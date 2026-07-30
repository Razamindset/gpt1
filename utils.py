"""
Shared helpers used by download_dataset.py, train_tokenizer.py, train.py and
generate.py:

  - Colab / Google Drive detection so every run persists its data,
    tokenizer, checkpoints and plots to Drive automatically -- nothing is
    lost if the Colab runtime disconnects.
  - A small TrainingLogger that writes step/epoch losses to CSV (so a
    resumed run keeps its history) and renders loss-curve PNGs.
  - Thin checkpoint save/load wrappers used by train.py and generate.py.
"""

import csv
import os
import sys

import torch


# --------------------------------------------------------------------------
# Paths / Google Drive
# --------------------------------------------------------------------------

def is_colab():
    """True when running inside a Google Colab runtime."""
    return "google.colab" in sys.modules


def mount_drive(drive_root="/content/drive"):
    """Mount Google Drive if on Colab. No-op (returns False) otherwise."""
    if not is_colab():
        return False

    from google.colab import drive  # noqa: import only exists on Colab

    if not os.path.ismount(drive_root):
        drive.mount(drive_root)
    return True


def setup_dirs(project_name="gpt1", drive_root="/content/drive/MyDrive", base_dir=None):
    """
    Returns a dict with the folders every script should read/write to:
        data, checkpoints, logs, plots

    On Colab this lives under Google Drive (survives disconnects).
    Locally (or if base_dir is given explicitly) it lives under ./runs/<project_name>.
    """
    if base_dir is not None:
        base = base_dir
    elif is_colab():
        mount_drive(drive_root="/content/drive")
        base = os.path.join(drive_root, project_name)
    else:
        base = os.path.join("runs", project_name)

    dirs = {
        "base": base,
        "data": os.path.join(base, "data"),
        "checkpoints": os.path.join(base, "checkpoints"),
        "logs": os.path.join(base, "logs"),
        "plots": os.path.join(base, "plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, epoch, global_step, val_loss, best_val_loss, config_snapshot=None):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": config_snapshot,
        },
        path,
    )


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


# --------------------------------------------------------------------------
# Training curve logging + plotting
# --------------------------------------------------------------------------

class TrainingLogger:
    """
    Tracks per-step training loss and per-epoch train/val loss + LR.
    Writes CSVs so history survives a Colab disconnect + resume, and can
    render a PNG loss-curve plot at any point.
    """

    def __init__(self, log_dir, plot_dir, resume=True):
        self.step_csv = os.path.join(log_dir, "step_history.csv")
        self.epoch_csv = os.path.join(log_dir, "epoch_history.csv")
        self.plot_path = os.path.join(plot_dir, "loss_curve.png")

        self.steps, self.step_losses = [], []
        self.epochs, self.train_losses, self.val_losses, self.lrs = [], [], [], []

        if resume:
            self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.step_csv):
            with open(self.step_csv, newline="") as f:
                for row in csv.DictReader(f):
                    self.steps.append(int(row["step"]))
                    self.step_losses.append(float(row["loss"]))
        if os.path.exists(self.epoch_csv):
            with open(self.epoch_csv, newline="") as f:
                for row in csv.DictReader(f):
                    self.epochs.append(int(row["epoch"]))
                    self.train_losses.append(float(row["train_loss"]))
                    self.val_losses.append(float(row["val_loss"]))
                    self.lrs.append(float(row["lr"]))

    def log_step(self, step, loss):
        self.steps.append(step)
        self.step_losses.append(loss)
        write_header = not os.path.exists(self.step_csv)
        with open(self.step_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "loss"])
            writer.writerow([step, loss])

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lrs.append(lr)
        write_header = not os.path.exists(self.epoch_csv)
        with open(self.epoch_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
            writer.writerow([epoch, train_loss, val_loss, lr])

    def plot(self, show=False):
        """Render step-loss + epoch train/val loss + LR schedule to a PNG."""
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Smoothed per-step training loss (the "is it learning" curve)
        ax = axes[0]
        if self.steps:
            ax.plot(self.steps, self.step_losses, color="tab:blue", alpha=0.3, linewidth=1, label="raw")
            smoothed = _moving_average(self.step_losses, window=max(1, len(self.step_losses) // 100))
            ax.plot(self.steps, smoothed, color="tab:blue", linewidth=2, label="smoothed")
            ax.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss (per step)")
        ax.grid(alpha=0.3)

        # Train vs val loss per epoch
        ax = axes[1]
        if self.epochs:
            ax.plot(self.epochs, self.train_losses, marker="o", ms=3, label="train")
            ax.plot(self.epochs, self.val_losses, marker="o", ms=3, label="val")
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train / Val Loss (per epoch)")
        ax.grid(alpha=0.3)

        # LR schedule
        ax = axes[2]
        if self.epochs:
            ax.plot(self.epochs, self.lrs, color="tab:green", marker="o", ms=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        return self.plot_path


def _moving_average(values, window=10):
    if window <= 1:
        return values
    out = []
    running_sum = 0.0
    q = []
    for v in values:
        q.append(v)
        running_sum += v
        if len(q) > window:
            running_sum -= q.pop(0)
        out.append(running_sum / len(q))
    return out
