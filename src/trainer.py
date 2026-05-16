"""
src/trainer.py
Implementa el loop de entrenamiento y evaluación.
No calcula métricas de desempeño (eso es metrics.py)
ni eficiencia (eso es efficiency.py).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.config import DatasetConfig, ExperimentConfig
from src.metrics import compute_classification_metrics


# ─── Tipos ────────────────────────────────────────────────────────────────────

TrainHistory = Dict[str, List]  # {"train_loss_by_step": [...], "val_loss_by_epoch": [...]}


# ─── Entrenamiento ────────────────────────────────────────────────────────────

def train_and_evaluate(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    exp_cfg: ExperimentConfig,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    checkpoint_callback=None,   # ← NUEVO: DriveCheckpointer o None
    resume_state: Optional[Dict] = None,  # ← NUEVO: estado previo o None
) -> Tuple[nn.Module, TrainHistory]:
    """
    Loop completo de entrenamiento + evaluación en validation.

    Args:
        checkpoint_callback: instancia de DriveCheckpointer, o None para
                             deshabilitar checkpointing (comportamiento original).
        resume_state:        dict devuelto por DriveCheckpointer.load(), o None
                             para empezar desde la época 1.

    Returns:
        (model entrenado, historial de curvas de loss, training_time)
    """
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Si resume_state contiene un optimizer ya cargado desde el checkpoint,
    # lo reutilizamos (preserva los momentos de Adam). Si no, creamos uno nuevo.
    if resume_state is not None and "optimizer" in resume_state:
        optimizer = resume_state["optimizer"]
        print("[trainer] Reutilizando optimizer desde checkpoint.")
    else:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=exp_cfg.learning_rate,
            weight_decay=exp_cfg.weight_decay,
        )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader) // exp_cfg.gradient_accumulation_steps
    total_steps = steps_per_epoch * exp_cfg.num_epochs
    warmup_steps = int(0.1 * total_steps)

    if resume_state is not None and "scheduler" in resume_state:
        scheduler = resume_state["scheduler"]
        print("[trainer] Reutilizando scheduler desde checkpoint.")
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # ── Mixed precision ────────────────────────────────────────────────────────
    use_fp16 = exp_cfg.fp16 and device.type == "cuda"
    if resume_state is not None and "scaler" in resume_state:
        scaler = resume_state["scaler"]
        print("[trainer] Reutilizando GradScaler desde checkpoint.")
    else:
        scaler = GradScaler() if use_fp16 else None

    print(f"\n[trainer] fp16={'ON' if use_fp16 else 'OFF'}, "
          f"gradient_accumulation_steps={exp_cfg.gradient_accumulation_steps}")
    print(f"[trainer] total_steps={total_steps}, warmup_steps={warmup_steps}")
    print(f"[trainer] Iniciando entrenamiento por {exp_cfg.num_epochs} épocas...\n")

    # ── Reanudar desde checkpoint si se provee ─────────────────────────────────
    if resume_state is not None:
        history     = resume_state["history"]
        global_step = resume_state["global_step"]
        start_epoch = resume_state["epoch"] + 1
    else:
        history: TrainHistory = {
            "train_loss_by_step": [],
            "val_loss_by_epoch": [],
        }
        global_step = 0
        start_epoch = 1

    accum_loss = 0.0
    accum_steps_count = 0
    training_start = time.time()

    model.to(device)
    model.train()

    for epoch in range(start_epoch, exp_cfg.num_epochs + 1):
        print(f"─── Época {epoch}/{exp_cfg.num_epochs} ──────────────────────────")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            if use_fp16:
                with autocast():
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
                    loss = output.loss / exp_cfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)
                loss = output.loss / exp_cfg.gradient_accumulation_steps
                loss.backward()

            accum_loss += loss.item()
            accum_steps_count += 1

            # Optimizer step cada gradient_accumulation_steps batches
            if (batch_idx + 1) % exp_cfg.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Registrar training loss
                avg_loss = accum_loss / accum_steps_count
                history["train_loss_by_step"].append({
                    "step": global_step,
                    "loss": round(avg_loss * exp_cfg.gradient_accumulation_steps, 6),
                })
                accum_loss = 0.0
                accum_steps_count = 0

                # Logging periódico
                if global_step % exp_cfg.logging_steps == 0:
                    print(f"  step={global_step:5d} | train_loss={avg_loss * exp_cfg.gradient_accumulation_steps:.4f}")

        # ── Evaluación al final de cada época ─────────────────────────────────
        val_loss = _evaluate_loss(model, val_loader, device, use_fp16)
        history["val_loss_by_epoch"].append({
            "epoch": epoch,
            "loss": round(val_loss, 6),
        })
        print(f"  → val_loss={val_loss:.4f}")

        # ── Checkpoint al final de cada época (si está habilitado) ────────────
        if checkpoint_callback is not None:
            checkpoint_callback.save(
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                scaler=scaler,
            )

    training_time = time.time() - training_start
    print(f"\n[trainer] Entrenamiento completado en {training_time:.1f}s")

    return model, history, training_time


# ─── Evaluación final ─────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool = False,
) -> Dict:
    """
    Evaluación completa: recolecta y_true, y_pred y loss sobre todo el split.
    Devuelve dict con las listas y el loss promedio.
    """
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_fp16:
                with autocast():
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
            else:
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)

            preds = output.logits.argmax(dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            if output.loss is not None:
                total_loss += output.loss.item()
                n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return {
        "y_true": all_labels,
        "y_pred": all_preds,
        "loss": avg_loss,
    }


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> float:
    """Calcula solo el val_loss durante el entrenamiento (sin recopilar preds)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_fp16:
                with autocast():
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=labels)
            else:
                output = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels)

            if output.loss is not None:
                total_loss += output.loss.item()
                n_batches += 1

    model.train()
    return total_loss / n_batches if n_batches > 0 else 0.0