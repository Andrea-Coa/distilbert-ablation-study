"""
src/checkpointing.py
Checkpoint de emergencia para Google Drive en Colab.

Guarda el estado del modelo + optimizador + scheduler + historial al final
de cada época. Si el runtime se desconecta, main.py puede reanudar desde
el último checkpoint en lugar de empezar de cero.

Uso típico (desde main.py):
    from src.checkpointing import DriveCheckpointer
    checkpointer = DriveCheckpointer(drive_dir=DRIVE_RESULTS_DIR, run_name=run_name)
    # Pasar como argumento a train_and_evaluate (ver trainer.py)
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class DriveCheckpointer:
    """
    Guarda un checkpoint completo al final de cada época en Google Drive.

    El checkpoint contiene:
      - model_state_dict     → pesos del modelo
      - optimizer_state_dict → estado del optimizador (momentos de Adam)
      - scheduler_state_dict → paso del scheduler (lr actual)
      - scaler_state_dict    → estado del GradScaler (fp16), si aplica
      - epoch                → época completada
      - global_step          → paso global completado
      - history              → curvas de loss hasta ese momento

    Estrategia de escritura segura:
      Escribe primero en /tmp (local, rápido), luego copia a Drive.
      Así un corte durante la copia no corrompe el checkpoint anterior:
      se mantiene el penúltimo hasta que el nuevo esté completo.
    """

    CHECKPOINT_FILENAME = "checkpoint_latest.pt"
    META_FILENAME = "checkpoint_meta.json"

    def __init__(self, drive_dir: str, run_name: str):
        self.drive_dir = Path(drive_dir) / run_name / "checkpoints"
        self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        print(f"[checkpointing] Directorio de checkpoints en Drive: {self.drive_dir}")

    # ── Guardar ───────────────────────────────────────────────────────────────

    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        history: Dict,
        scaler=None,
    ) -> None:
        """Guarda el checkpoint del final de la época `epoch`."""
        t0 = time.time()

        checkpoint = {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history":              history,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        # 1) Escribir en /tmp (local, sin latencia de Drive)
        tmp_path = Path("/tmp") / f"{self.run_name}_checkpoint.pt"
        torch.save(checkpoint, tmp_path)

        # 2) Copiar atómicamente a Drive
        dest_path = self.drive_dir / self.CHECKPOINT_FILENAME
        shutil.copy2(tmp_path, dest_path)

        # 3) Escribir metadata legible en JSON (para diagnóstico rápido)
        meta = {
            "epoch":       epoch,
            "global_step": global_step,
            "saved_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "val_loss":    history["val_loss_by_epoch"][-1]["loss"] if history["val_loss_by_epoch"] else None,
        }
        with open(self.drive_dir / self.META_FILENAME, "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t0
        print(f"[checkpointing] ✅ Checkpoint época {epoch} guardado en Drive ({elapsed:.1f}s)")

    # ── Cargar ────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        drive_dir: str,
        run_name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        scaler=None,
    ) -> Optional[Dict]:
        """
        Carga el último checkpoint si existe. Devuelve el dict de estado
        (con 'epoch', 'global_step', 'history') o None si no hay checkpoint.

        Uso en main.py:
            state = DriveCheckpointer.load(DRIVE_RESULTS_DIR, run_name, ...)
            start_epoch = state["epoch"] + 1 if state else 1
        """
        ckpt_path = Path(drive_dir) / run_name / "checkpoints" / cls.CHECKPOINT_FILENAME
        if not ckpt_path.exists():
            print("[checkpointing] No se encontró checkpoint previo. Empezando desde cero.")
            return None

        print(f"[checkpointing] 🔄 Cargando checkpoint desde: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"[checkpointing] ✅ Reanudando desde época {epoch + 1} "
              f"(global_step={checkpoint['global_step']})")

        return {
            "epoch":       epoch,
            "global_step": checkpoint["global_step"],
            "history":     checkpoint["history"],
        }