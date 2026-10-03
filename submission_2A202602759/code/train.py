"""train.py — Pipeline huấn luyện, đánh giá và xuất dự đoán.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val trong notebook, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1)
                      for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    confusion = torch.zeros((7, 7), dtype=torch.int64, device=X.device)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name == "mse":
            per_sample = F.mse_loss(logits, F.one_hot(yb, 7).float(), reduction="none").mean(dim=1)
            total_loss += per_sample.sum().item()
        else:
            raise ValueError(f"loss không hợp lệ: {loss_name}")
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        confusion += torch.bincount(7 * yb + pred, minlength=49).reshape(7, 7)
    return {"loss": total_loss / len(X), "acc": correct / len(X),
            "macro_f1": macro_f1_from_confusion(confusion.cpu().numpy())}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        # Trung bình trên 7 lớp rồi trung bình các mẫu trong batch.
        return F.mse_loss(logits, F.one_hot(y, 7).float())
    raise ValueError(f"loss không hợp lệ: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("Cần chọn lr bằng validation trước khi chạy thí nghiệm")
    set_seed(int(cfg["seed"]))
    device = data["X_tr"].device
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                cfg["weight_decay"], cfg["momentum"])
    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 chỉ hỗ trợ CUDA trong lab này")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step0_loss = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])["loss"]
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_acc",
                                   "val_macro_f1", "grad_norm", "epoch_time_s")}
    best_state = None
    best_epoch = 0
    best_val_loss = math.inf
    best_metrics = None
    diverged = False
    # Cùng seed tạo cùng thứ tự xáo lô cho cùng cấu hình và thiết bị.
    generator = torch.Generator(device=device.type).manual_seed(int(cfg["seed"]))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model.train()
        grad_norms = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            if precision == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            norm = clip_gradients(model.parameters(), cfg["clip_norm"])
            if not math.isfinite(norm):
                diverged = True
                break
            grad_norms.append(norm)
            if precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        if diverged:
            break
        train_metrics = evaluate(model, data["X_tr"], data["y_tr"], cfg["loss"])
        val_metrics = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        for key, value in (("epoch", epoch), ("train_loss", train_metrics["loss"]),
                           ("val_loss", val_metrics["loss"]), ("val_acc", val_metrics["acc"]),
                           ("val_macro_f1", val_metrics["macro_f1"]),
                           ("grad_norm", float(np.mean(grad_norms))), ("epoch_time_s", elapsed)):
            history[key].append(value)
        if not math.isfinite(val_metrics["loss"]):
            diverged = True
            break
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = (train_metrics, val_metrics)
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if best_state is not None else None,
        "best_epoch": best_epoch or None,
        "final_train_loss": history["train_loss"][-1] if history["epoch"] else None,
        "final_val_loss": history["val_loss"][-1] if history["epoch"] else None,
        "val_acc": best_metrics[1]["acc"] if best_metrics else None,
        "val_macro_f1": best_metrics[1]["macro_f1"] if best_metrics else None,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch"] else None,
        "peak_mem_MB": (torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None),
        "diverged": diverged,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if len(row_id) != len(preds) or len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id và pred phải đủ dòng và row_id không trùng")
    if not np.isin(preds, range(7)).all():
        raise ValueError("pred phải nằm trong 0..6")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_id, preds))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    if result["best_state"] is None:
        raise ValueError("Thí nghiệm phân kỳ, không có best_state để dự đoán eval")
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"])
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]
    model.load_state_dict(result["best_state"])
    model = model.to(data["X_eval"].device)
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions.cpu().numpy(), pred_path)
