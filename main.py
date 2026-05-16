"""
main.py
Punto de entrada del pipeline de ablation study DistilBERT.
Orquesta: config → datos → modelo → entrenamiento → evaluación → guardado.

Cambios respecto al original:
  - Importa DriveCheckpointer desde src/checkpointing.py
  - Detecta DRIVE_RESULTS_DIR en el entorno (seteado en el notebook de Colab)
  - Intenta reanudar desde el último checkpoint antes de entrenar
  - Pasa checkpoint_callback y resume_state a train_and_evaluate
  - Todo lo demás es idéntico al original
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.checkpointing import DriveCheckpointer
from src.config import load_dataset_config, load_experiment_config
from src.data import load_data
from src.efficiency import count_parameters, measure_gpu_memory, measure_latency
from src.freezing import apply_freeze_strategy
from src.metrics import compute_classification_metrics
from src.model import build_model
from src.reporting import (
    append_to_results_csv,
    build_result_dict,
    print_summary,
    save_config_copy,
    save_loss_curves_json,
    save_loss_curves_plot,
    save_result_json,
)
from src.trainer import evaluate, train_and_evaluate
from src.utils import empty_cuda_cache, get_device, make_output_dirs, make_run_name, set_seed

import torch


def main():
    parser = argparse.ArgumentParser(description="DistilBERT Ablation Study")
    parser.add_argument("--dataset",    required=True, help="Path al YAML del dataset")
    parser.add_argument("--experiment", required=True, help="Path al YAML del experimento")
    parser.add_argument("--output_dir", default="outputs", help="Directorio base de salida")
    args = parser.parse_args()

    # ── Configuración ─────────────────────────────────────────────────────────
    dataset_cfg = load_dataset_config(args.dataset)
    exp_cfg     = load_experiment_config(args.experiment)

    run_name = make_run_name(dataset_cfg.output_dataset_name, exp_cfg.experiment_name)
    dirs = make_output_dirs(args.output_dir)

    # ── DRIVE: directorio de checkpoints ──────────────────────────────────────
    # DRIVE_RESULTS_DIR se define en la celda del notebook antes de ejecutar main.py:
    #   DRIVE_RESULTS_DIR = "/content/drive/MyDrive/distilbert_ablation_results"
    drive_dir = os.environ.get("DRIVE_RESULTS_DIR", "")
    checkpointer = DriveCheckpointer(drive_dir, run_name) if drive_dir else None
    if not drive_dir:
        print("[main] ⚠ DRIVE_RESULTS_DIR no definido — checkpointing deshabilitado.")

    # ── FASE DE REPRODUCIBILIDAD ──────────────────────────────────────────────
    set_seed(exp_cfg.seed)
    empty_cuda_cache()

    # ── DISPOSITIVO ───────────────────────────────────────────────────────────
    device = get_device()

    # ── FASE DE DATOS ─────────────────────────────────────────────────────────
    loaders = load_data(dataset_cfg, exp_cfg)

    # ── FASE DE MODELO ────────────────────────────────────────────────────────
    print(f"\n[main] Construyendo modelo...")
    model = build_model(exp_cfg, dataset_cfg)
    apply_freeze_strategy(model, exp_cfg)

    # ── REANUDAR DESDE CHECKPOINT (si existe) ─────────────────────────────────
    # Para reanudar necesitamos el optimizer/scheduler ya construidos, pero
    # train_and_evaluate los construye internamente. Delegamos la carga al
    # propio trainer pasando resume_state como dict de historia + step + epoch.
    # Si existe un checkpoint, DriveCheckpointer.load() reconstruye el estado
    # completo DENTRO de train_and_evaluate (ver trainer.py).
    #
    # Flujo simplificado que usamos aquí: pasamos el checkpoint_path al trainer
    # y dejamos que él decida si reanudar. Solo necesitamos saber si hay un
    # checkpoint para no repetir trabajo ya hecho.
    resume_state = None
    if checkpointer is not None:
        # Construir temporalmente optimizer/scheduler para poder cargar el estado
        _optimizer_tmp = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=exp_cfg.learning_rate,
            weight_decay=exp_cfg.weight_decay,
        )
        _steps_per_epoch = len(loaders["train"]) // exp_cfg.gradient_accumulation_steps
        _total_steps     = _steps_per_epoch * exp_cfg.num_epochs
        _warmup_steps    = int(0.1 * _total_steps)
        from transformers import get_linear_schedule_with_warmup
        _scheduler_tmp = get_linear_schedule_with_warmup(
            _optimizer_tmp,
            num_warmup_steps=_warmup_steps,
            num_training_steps=_total_steps,
        )
        use_fp16  = exp_cfg.fp16 and device.type == "cuda"
        _scaler_tmp = torch.cuda.amp.GradScaler() if use_fp16 else None

        resume_state = DriveCheckpointer.load(
            drive_dir=drive_dir,
            run_name=run_name,
            model=model,
            optimizer=_optimizer_tmp,
            scheduler=_scheduler_tmp,
            device=device,
            scaler=_scaler_tmp,
        )
        # Nota: si resume_state is not None, el modelo ya tiene los pesos cargados.
        # train_and_evaluate recibirá resume_state y NO re-creará optimizer/scheduler
        # desde cero; usará los que ya están cargados en el modelo.
        # El optimizer/scheduler pre-cargados los pasamos en resume_state para
        # que trainer.py los reutilice en lugar de crear nuevos.
        if resume_state is not None:
            resume_state["optimizer"] = _optimizer_tmp
            resume_state["scheduler"] = _scheduler_tmp
            resume_state["scaler"]    = _scaler_tmp

    # ── FASE DE ENTRENAMIENTO ─────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, history, training_time = train_and_evaluate(
        model=model,
        loaders=loaders,
        exp_cfg=exp_cfg,
        dataset_cfg=dataset_cfg,
        device=device,
        checkpoint_callback=checkpointer,
        resume_state=resume_state,
    )

    # ── FASE DE EVALUACIÓN ────────────────────────────────────────────────────
    use_fp16 = exp_cfg.fp16 and device.type == "cuda"
    print("\n[main] Evaluando en test set...")
    test_results = evaluate(model, loaders["test"], device, use_fp16=use_fp16)

    performance_metrics = compute_classification_metrics(
        y_true=test_results["y_true"],
        y_pred=test_results["y_pred"],
        loss_value=test_results["loss"],
    )

    print(f"[main] accuracy={performance_metrics['accuracy']:.4f} | "
          f"f1_macro={performance_metrics['f1_macro']:.4f}")

    # ── FASE DE EFICIENCIA ────────────────────────────────────────────────────
    print("\n[main] Midiendo métricas de eficiencia...")
    param_metrics   = count_parameters(model)
    latency_metrics = measure_latency(model, loaders["test"], device)
    gpu_metrics     = measure_gpu_memory()
    efficiency_metrics = {**param_metrics, **latency_metrics, **gpu_metrics}

    # ── FASE DE GUARDADO ──────────────────────────────────────────────────────
    loss_curves_path = dirs["logs"]    / f"{run_name}_loss_curves.json"
    loss_plot_path   = dirs["plots"]   / f"{run_name}_loss_curves.png"
    result_json_path = dirs["results"] / f"{run_name}.json"
    model_save_path  = dirs["models"]  / run_name
    config_save_dir  = dirs["results"]

    save_loss_curves_json(history, loss_curves_path)
    save_loss_curves_plot(
        history, loss_plot_path,
        title=f"{dataset_cfg.output_dataset_name} | {exp_cfg.experiment_name}"
    )

    result = build_result_dict(
        dataset_cfg=dataset_cfg,
        exp_cfg=exp_cfg,
        performance_metrics=performance_metrics,
        efficiency_metrics=efficiency_metrics,
        training_time=training_time,
        run_name=run_name,
        loss_curves_path=loss_curves_path,
        model_path=model_save_path,
    )
    save_result_json(result, result_json_path)
    append_to_results_csv(result)

    model_save_path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(str(model_save_path / "backbone"))
    torch.save(model.head.state_dict(), str(model_save_path / "head.pt"))
    print(f"[main] Modelo guardado: {model_save_path}")

    save_config_copy(args.dataset, args.experiment, config_save_dir, run_name)

    # ── COPIAR RESULTADOS FINALES A DRIVE ─────────────────────────────────────
    if drive_dir:
        import shutil
        from pathlib import Path as _Path
        drive_run_dir = _Path(drive_dir) / run_name
        drive_run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(result_json_path, drive_run_dir / result_json_path.name)
        shutil.copy2(loss_curves_path, drive_run_dir / loss_curves_path.name)
        shutil.copy2(loss_plot_path,   drive_run_dir / loss_plot_path.name)
        print(f"[main] ✅ Resultados finales copiados a Drive: {drive_run_dir}")

    print_summary(result)
    print(f"[main] ✓ Experimento completado. Run: {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())