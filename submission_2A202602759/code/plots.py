"""plots.py — Vẽ đường cong huấn luyện và so sánh thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError:  # Môi trường CPU tối giản có Pillow nhưng không có Matplotlib.
    plt = None
from pathlib import Path


def _pil_lines(panels, title: str, path: str) -> None:
    """Vẽ PNG dự phòng bằng Pillow khi môi trường không cài Matplotlib."""
    from PIL import Image, ImageDraw

    width, height = 500 * len(panels), 410
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), title.encode("ascii", "replace").decode("ascii"), fill="black")
    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    for panel_index, (label, series) in enumerate(panels):
        x0 = 35 + panel_index * 500
        y0, x1, y1 = 65, x0 + 420, 350
        draw.rectangle((x0, y0, x1, y1), outline="black")
        draw.text((x0, 44), label, fill="black")
        values = [float(value) for _, points in series for value in points]
        if not values:
            continue
        low, high = min(values), max(values)
        if high == low:
            high += 1
        draw.text((x0, y1 + 5), "Epoch / step", fill="black")
        draw.text((x0 + 4, y0 + 4), f"{high:.3g}", fill="black")
        draw.text((x0 + 4, y1 - 15), f"{low:.3g}", fill="black")
        for series_index, (name, points) in enumerate(series):
            if not points:
                continue
            color = palette[series_index % len(palette)]
            coords = [(x0 + 10 + (x1 - x0 - 20) * j / max(1, len(points) - 1),
                       y1 - 12 - (y1 - y0 - 24) * (float(value) - low) / (high - low))
                      for j, value in enumerate(points)]
            if len(coords) > 1:
                draw.line(coords, fill=color, width=3)
            else:
                draw.ellipse((coords[0][0]-2, coords[0][1]-2, coords[0][0]+2, coords[0][1]+2), fill=color)
            draw.text((x0 + 135 * (series_index % 3), 370 + 16 * (series_index // 3)),
                      name[:20], fill=color)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def plot_health(losses, path: str) -> None:
    """Vẽ loss khi quá khớp 20 mẫu."""
    if plt is None:
        _pil_lines([("CE loss", [("train 20", losses)])], "Overfit 20 samples", path)
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set(xlabel="Step", ylabel="CE loss", title="Overfit 20 training samples")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    history = result["history"]
    cfg = result["cfg"]
    epochs = history["epoch"]
    if plt is None:
        _pil_lines([
            ("Loss", [("train", history["train_loss"]), ("val", history["val_loss"])]),
            ("Validation", [("accuracy", history["val_acc"]), ("macro-F1", history["val_macro_f1"])]),
            ("Grad norm before clip", [("grad norm", history["grad_norm"])])],
            f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']}", path)
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["val_acc"], label="val accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axes[1].set_title("Validation metrics")
    axes[2].plot(epochs, history["grad_norm"], label="grad norm before clipping")
    axes[2].set_title("Gradient norm")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.3)
        if result["summary"]["best_epoch"]:
            axis.axvline(result["summary"]["best_epoch"], color="gray", ls="--", alpha=0.5)
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} {cfg['precision']}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if plt is None:
        _pil_lines([(metric, [(r["cfg"]["exp_id"], r["history"][metric]) for r in results])],
                   title or f"So sánh {metric}", path)
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise KeyError(f"metric không có trong history: {metric}")
        axis.plot(history["epoch"], history[metric], label=result["cfg"]["exp_id"])
    axis.set(xlabel="Epoch", ylabel=metric, title=title or f"So sánh {metric}")
    axis.legend()
    axis.grid(alpha=0.3)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
