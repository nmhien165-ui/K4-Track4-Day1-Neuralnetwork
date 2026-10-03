"""results_table.py — Ghi JSON và điền bảng thí nghiệm từ mẫu.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    target = Path(results_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{result['cfg']['exp_id']}.json"
    payload = {key: result[key] for key in ("cfg", "history", "summary")}
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    results = []
    for path in sorted(Path(results_dir).glob("*.json")):
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
        if {"cfg", "history", "summary"}.issubset(result):
            results.append(result)
    return sorted(results, key=lambda result: result["cfg"]["exp_id"])


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg, summary = result["cfg"], result["summary"]
    row = {**cfg, **summary}
    row["hidden"] = "-".join(map(str, cfg["hidden"]))
    row["clip_norm"] = cfg["clip_norm"] if cfg["clip_norm"] is not None else "none"
    row["figure_file"] = f"figures/{cfg['exp_id']}.png"
    row["notes"] = notes
    if eval_scores is not None:
        row["eval_acc"] = eval_scores["accuracy"]
        row["eval_macro_f1"] = eval_scores["macro_f1"]
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    import openpyxl
    from copy import copy
    from openpyxl.formula.translate import Translator

    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    headers = {ws.cell(1, col).value: col for col in range(1, ws.max_column + 1)}
    formula_names = ("step0_gap_vs_lnC", "gap_val_minus_train",
                     "delta_val_f1_vs_base", "beyond_noise")
    # Xoá ví dụ ở dòng 2 và dữ liệu cũ, giữ định dạng cùng các cột công thức.
    for line in range(2, max(ws.max_row, len(rows) + 1) + 1):
        for name, col in headers.items():
            if name not in formula_names:
                ws.cell(line, col).value = None
    for line, row in enumerate(rows, start=2):
        for name, value in row.items():
            if name in headers and name not in formula_names:
                ws.cell(line, headers[name]).value = value
        if line > ws.max_row:
            for col in range(1, ws.max_column + 1):
                ws.cell(line, col)._style = copy(ws.cell(2, col)._style)
        for name in formula_names:
            col = headers[name]
            first = ws.cell(2, col)
            target = ws.cell(line, col)
            if target.value is None and first.data_type == "f":
                target.value = Translator(first.value, origin=first.coordinate).translate_formula(target.coordinate)
    seeds = wb["Seeds"]
    baseline_ids = [row["exp_id"] for row in rows if row.get("group") == "baseline"]
    for line in range(2, 7):
        seeds.cell(line, 1).value = baseline_ids[line - 2] if line - 2 < len(baseline_ids) else None
    summary = wb["Summary"]
    for line in range(2, 12):
        group = summary.cell(line, 1).value
        group_rows = [row for row in rows if row.get("group") == group]
        if group_rows:
            best = max((row for row in group_rows if row.get("val_macro_f1") is not None),
                       key=lambda row: row["val_macro_f1"], default=None)
            summary.cell(line, 8).value = (f"Cao nhất: {best['exp_id']} (val macro-F1="
                                           f"{best['val_macro_f1']:.4f})" if best else "Chạy phân kỳ")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def write_report(rows: list[dict], baseline_eval: dict, final_eval: dict,
                 final_exp_id: str, health: dict, path: str) -> None:
    """Tạo báo cáo từ số đo thật sau khi notebook chạy xong; không tự tạo số liệu."""
    import math
    import statistics

    by_id = {row["exp_id"]: row for row in rows}
    baseline = by_id["base-s1"]
    baseline_rows = [row for row in rows if row.get("group") == "baseline"]
    baseline_f1 = [row["val_macro_f1"] for row in baseline_rows if row.get("val_macro_f1") is not None]
    baseline_acc = [row["val_acc"] for row in baseline_rows if row.get("val_acc") is not None]
    sigma = statistics.stdev(baseline_f1) if len(baseline_f1) >= 2 else None
    noise = 2 * sigma if sigma is not None else None
    chosen = by_id[final_exp_id]

    def metric(value):
        if value is None:
            return "chưa đo"
        return f"{value:.2e}" if 0 < abs(value) < 0.001 else f"{value:.4f}"

    lines = [
        "# Báo cáo Lab Day 1 — 2A202602759", "",
        "## 1. Thiết lập", "",
        f"- Môi trường: Google Colab, thiết bị {health['device']}; PyTorch {health['torch_version']}.",
        "- Dữ liệu: Forest CoverType, 464.809 train và 116.203 eval theo metadata cố định. "
        "Tách 20% train làm validation (seed 42, phân tầng): 371.847 train, 92.962 val. "
        "Chỉ 10 cột số được chuẩn hóa bằng thống kê từ phần train sau khi tách.",
        f"- Baseline: M-base 54→256→128→7, CE, SGD+momentum 0,9, lr={baseline['lr']}, "
        "batch 512, 20 epoch, He ở lớp ẩn và He×0,1 ở lớp logit, FP32; chọn lr từ validation.",
        f"- Accuracy đoán lớp đa số trên val: {metric(health['majority_acc'])}.", "",
        "- Chủ đề đã thử: loss, optimizer, hyper-parameter (độ rộng), dropout, "
        "clipping, mixed precision (nếu có GPU) và khởi tạo tham số.", "",
        "## 2. Kiểm tra ban đầu và độ nhiễu", "",
        "| Kiểm tra | Kết quả |", "|---|---|",
        f"| Số tham số / shape logits | {health['n_params']} / {health['logits_shape']} |",
        f"| Loss bước 0 (so với ln 7 = {math.log(7):.4f}) | {metric(health['step0_loss'])} |",
        f"| Quá khớp 20 mẫu | loss cuối {metric(health['small_loss'])}, "
        f"accuracy {metric(health['small_acc'])} |",
        f"| Mọi tham số có gradient khác 0 | {health['gradients_ok']} |",
        f"| Baseline, số seed đã chạy | {len(baseline_f1)} |",
        f"| Baseline val accuracy (TB ± σ) | "
        f"{metric(statistics.mean(baseline_acc) if baseline_acc else None)} ± "
        f"{metric(statistics.stdev(baseline_acc) if len(baseline_acc) >= 2 else None)} |",
        f"| Baseline val macro-F1 (TB ± σ) | "
        f"{metric(statistics.mean(baseline_f1) if baseline_f1 else None)} ± {metric(sigma)} |",
        "", "![Loss khi quá khớp 20 mẫu](figures/compare_health_overfit20.png)", "",
        f"**Ngưỡng nhiễu 2σ của val macro-F1:** {metric(noise)}. "
        "Chỉ hai seed nên ước lượng này còn thô.", "",
        "## 3. Kết quả theo chủ đề", "",
        "Mỗi dòng dưới đây trỏ về sheet `Experiments` và ảnh riêng trong `figures/`. "
        "Đổi một yếu tố, ngoại trừ optimizer cần chỉnh lr riêng để so công bằng. "
        "CE và MSE được so bằng macro-F1, không so giá trị loss khác thang đo.", "",
        "| Chủ đề | exp_id | Dự đoán trước | val macro-F1 | Δ so với base-s1 | Vượt 2σ? | Ảnh |",
        "|---|---|---|---:|---:|---|---|",
    ]
    diagnostics = []
    if health.get("init_diagnostics"):
        diagnostics = ["- Kiểm tra bước 0: độ lệch chuẩn sau từng Linear/ReLU trên 512 mẫu val; "
                       "`zeros` và `normal` chưa huấn luyện.", "",
                       "| Init | Loss bước 0 | Std kích hoạt theo lớp |",
                       "|---|---:|---|",]
        for name, measured in health["init_diagnostics"].items():
            values = ", ".join(f"{value:.3f}" for value in measured["activation_std"])
            diagnostics.append(f"| {name} | {metric(measured['step0_val_loss'])} | {values} |")
        diagnostics.append("")
    for row in rows:
        if row["group"] == "baseline":
            continue
        value = row.get("val_macro_f1")
        delta = None if value is None else value - baseline["val_macro_f1"]
        significant = "chưa đo" if delta is None or noise is None else ("có" if abs(delta) > noise else "không")
        prediction = str(row.get("description", "")).replace("|", "/")
        lines.append(f"| {row['group']} | `{row['exp_id']}` | {prediction} | {metric(value)} | "
                     f"{metric(delta)} | {significant} | [đường cong](figures/{row['exp_id']}.png) |")
    def noise_note(row: dict) -> str:
        delta = row["val_macro_f1"] - baseline["val_macro_f1"]
        return "vượt 2σ" if noise is not None and abs(delta) > noise else "dưới 2σ hoặc chưa đo nhiễu"

    if "loss-mse" in by_id:
        row = by_id["loss-mse"]
        lines += ["", "### 3.1 Hàm mất mát — CE vs MSE", "",
                  "- Dự đoán: MSE trên one-hot có thể kém CE do gradient phân loại khác nhau.",
                  f"- Kết quả: `loss-mse` {metric(row['val_macro_f1'])} so với CE `base-s1` "
                  f"{metric(baseline['val_macro_f1'])} ({noise_note(row)}). "
                  "[Ảnh MSE](figures/loss-mse.png) và [CE](figures/base-s1.png).",
                  "- Giải thích: CE tối ưu xác suất lớp đúng; MSE có thể cho gradient nhỏ khi softmax "
                  "bão hòa. Không so trực tiếp giá trị hai loss khác thang đo."]
    optimizer_rows = [row for row in rows if row.get("group") == "optimizer"]
    if optimizer_rows:
        lines += ["", "### 3.2 Bộ tối ưu hóa", "",
                  "- Dự đoán: Adam có thể học nhanh hơn SGD momentum khi LR phù hợp.",
                  "| exp_id | Optimizer | LR | Best epoch | Val macro-F1 |",
                  "|---|---|---:|---:|---:|"]
        for row in [baseline] + optimizer_rows:
            lines.append(f"| `{row['exp_id']}` | {row['optimizer']} | {row['lr']} | "
                         f"{row['best_epoch']} | {metric(row['val_macro_f1'])} |")
        lines += ["", "LR 0,001 và 0,003 của Adam, LR 0,05 và 0,1 của SGD cho kết quả khác nhau; "
                  "chỉ so mỗi optimizer ở LR tốt nhất đã thử. Adam chia bước theo moment của từng tham số. "
                  "[Ảnh chồng optimizer](figures/compare_optimizer.png)."]
    if "wide" in by_id:
        row = by_id["wide"]
        lines += ["", "### 3.3 Hyper-parameter — độ rộng", "",
                  "- Dự đoán: mạng rộng có thể cải thiện F1 với chi phí lớn hơn.",
                  f"- `wide` {row['hidden']} đạt {metric(row['val_macro_f1'])} so với `base-s1` "
                  f"{metric(baseline['val_macro_f1'])} ({noise_note(row)}); "
                  f"thời gian/epoch {metric(row['time_per_epoch_s'])} so với "
                  f"{metric(baseline['time_per_epoch_s'])} giây. Cùng batch và epoch nên số bước cập nhật "
                  "không đổi. [Ảnh wide](figures/wide.png).",
                  "- Giải thích: độ rộng tăng năng lực biểu diễn lẫn chi phí tính toán; "
                  "chênh lệch dưới nhiễu chưa chứng minh lợi ích ổn định."]
    if "dropout-02" in by_id:
        row = by_id["dropout-02"]
        lines += ["", "### 3.4 Dropout", "",
                  "- Dự đoán: dropout 0,2 có thể giúp khi train–val gap thể hiện quá khớp.",
                  f"- `dropout-02` đạt {metric(row['val_macro_f1'])} so với `base-s1` "
                  f"{metric(baseline['val_macro_f1'])} ({noise_note(row)}). "
                  f"Loss train/val cuối: {metric(row['final_train_loss'])}/"
                  f"{metric(row['final_val_loss'])} so với {metric(baseline['final_train_loss'])}/"
                  f"{metric(baseline['final_val_loss'])}. [Ảnh dropout](figures/dropout-02.png).",
                  "- Giải thích: dropout giảm khả năng khớp train; gap nhỏ hơn cùng loss cao hơn "
                  "không chứng minh tổng quát tốt hơn."]
    if "clip-1" in by_id:
        row = by_id["clip-1"]
        lines += ["", "### 3.5 Gradient clipping", "",
                  "- Dự đoán: ngưỡng 1,0 có thể ít tác dụng nếu norm thường nhỏ.",
                  f"- `clip-1` đạt {metric(row['val_macro_f1'])} so với `base-s1` "
                  f"{metric(baseline['val_macro_f1'])} ({noise_note(row)}). "
                  "[Ảnh clipping](figures/clip-1.png).",
                  "- Giải thích: ảnh chỉ ghi norm trước clip tổng hợp theo epoch, chưa đếm số "
                  "bước bị clip. Chưa thử cặp LR cao có/không clip nên chưa chứng minh nó cứu huấn luyện."]
    if "amp-fp16" in by_id:
        row = by_id["amp-fp16"]
        lines += ["", "### 3.6 Mixed precision", "",
                  "- Dự đoán: FP16 có thể nhanh/ít bộ nhớ hơn, nhưng MLP nhỏ có overhead.",
                  f"- `amp-fp16` vs FP32 `base-s1`: val macro-F1 {metric(row['val_macro_f1'])} "
                  f"vs {metric(baseline['val_macro_f1'])} ({noise_note(row)}); "
                  f"giây/epoch {metric(row['time_per_epoch_s'])} vs "
                  f"{metric(baseline['time_per_epoch_s'])}; đỉnh MB {metric(row['peak_mem_MB'])} "
                  f"vs {metric(baseline['peak_mem_MB'])}. [Ảnh FP16](figures/amp-fp16.png).",
                  "- Giải thích: autocast/GradScaler và kernel có thể tốn hơn lợi ích với mạng nhỏ. "
                  "BF16 chưa đo; một lần chạy chưa đủ khẳng định hiệu năng tổng quát."]
    if "init-xavier" in by_id:
        row = by_id["init-xavier"]
        lines += ["", "### 3.7 Khởi tạo tham số", "",
                  "- Dự đoán: He phù hợp ReLU; zeros giữ nơ-ron đối xứng, normal quá nhỏ làm tín hiệu co dần.",
                  *diagnostics,
                  f"- `init-xavier` đạt {metric(row['val_macro_f1'])} so với He `base-s1` "
                  f"{metric(baseline['val_macro_f1'])} ({noise_note(row)}). "
                  "[Ảnh Xavier](figures/init-xavier.png).",
                  "- Bảng khởi tạo bước 0 ở mục 2 là số đo kích hoạt; zeros/normal chưa được huấn luyện. "
                  "He có phương sai 2/fan-in, Xavier dùng cả fan-in và fan-out."]
    lines += ["", "**Đối chiếu và cơ chế:** Chênh lệch nhỏ hơn 2σ chưa đủ để kết luận một cấu hình tốt hơn. "
              "Đọc `figures/<exp_id>.png` để so tốc độ hội tụ, train–val gap và chuẩn gradient trước clip. "
              "CE và MSE tạo gradient khác nhau; dropout giảm đồng thích nghi nhưng có thể gây thiếu khớp; "
              "clipping chỉ thay đổi bước cập nhật khi chuẩn gradient vượt ngưỡng; "
              "He giữ phương sai qua ReLU tốt hơn khởi tạo quá nhỏ trên mạng đủ sâu.", "",
              "![So sánh optimizer trên validation](figures/compare_optimizer.png)", "",
              "![Đường cong baseline](figures/base-s1.png)", "",
              "## 4. Đánh giá cuối trên eval", "",
              "Cấu hình được chọn hoàn toàn theo val macro-F1 trong các lần chạy, "
              f"với epoch có val loss thấp nhất: `{final_exp_id}` (seed {chosen['seed']}).", "",
              "| Cấu hình | Seed nộp | val macro-F1 | eval macro-F1 | eval accuracy |",
              "|---|---:|---:|---:|---:|",
              f"| `base-s1` | {baseline['seed']} | {metric(baseline['val_macro_f1'])} | "
              f"{metric(baseline_eval['macro_f1'])} | {metric(baseline_eval['accuracy'])} |",
              f"| `{final_exp_id}` | {chosen['seed']} | {metric(chosen['val_macro_f1'])} | "
              f"{metric(final_eval['macro_f1'])} | {metric(final_eval['accuracy'])} |", "",
              f"Chênh lệch eval macro-F1 so với baseline: "
              f"{final_eval['macro_f1'] - baseline_eval['macro_f1']:+.4f}. "
              "Đây là đo sau khi khóa cấu hình theo val; không điều chỉnh lại từ eval. "
              f"Trên val, chênh lệch là {chosen['val_macro_f1'] - baseline['val_macro_f1']:+.4f} "
              f"so với 2σ={metric(noise)}. Không dùng ngưỡng nhiễu val để kết luận về eval; "
              "chưa đo độ nhiễu trực tiếp của eval.", "",
              "### 4.1 Phân tích lỗi theo lớp", "",
              "| Lớp | Support | Precision | Recall | F1 |", "|---:|---:|---:|---:|---:|",]
    for item in final_eval["per_class"]:
        lines.append(f"| {item['cls']} | {item['support']} | {item['precision']:.4f} | "
                     f"{item['recall']:.4f} | {item['f1']:.4f} |")
    matrix = final_eval["confusion_matrix"]
    hardest = min(final_eval["per_class"], key=lambda item: item["f1"])
    hard_id = hardest["cls"]
    confusions = list(matrix[hard_id]); confusions[hard_id] = -1
    confused_with = confusions.index(max(confusions))
    lines += ["", f"Lớp khó nhất là {hard_id} (F1={hardest['f1']:.4f}, support={hardest['support']}); "
              f"bị nhầm nhiều nhất với lớp {confused_with} ({matrix[hard_id][confused_with]} mẫu). "
              "Mất cân bằng và đặc trưng chồng lấn là hai giả thuyết cần kiểm tra thêm.", "",
              "Ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán):", "", "```text"]
    lines += [" ".join(f"{value:6d}" for value in matrix_row) for matrix_row in matrix]
    lines += ["```", "", "## 5. Trả lời các câu hỏi dẫn dắt", "",
              "1. **Optimizer nào thắng khi chỉnh LR công bằng?** Bảng ở mục 3.2 so mỗi loại "
              "ở LR tốt nhất đã thử. Adam 0,001 gần SGD 0,1, còn Adam 0,003 cao hơn; "
              "kết luận phụ thuộc LR và cần thêm seed Adam. [Ảnh](figures/compare_optimizer.png).",
              "2. **Dropout có giúp khi chưa quá khớp?** `dropout-02` cho loss train/val "
              "đều cao hơn baseline; gap nhỏ hơn không tự chứng minh tổng quát tốt hơn. "
              "Chỉ nên cân nhắc khi train tốt lên mà val xấu đi. [Ảnh](figures/dropout-02.png).",
              "3. **Clipping giải quyết gì và đã chứng minh chưa?** Nó giới hạn norm gradient "
              "từng bước; log theo epoch chưa cho biết số bước chạm ngưỡng và chưa thử LR cao, "
              "nên thí nghiệm hiện tại chưa chứng minh nó cứu huấn luyện. [Ảnh](figures/clip-1.png).",
              ("4. Trên GPU T4, FP16 có thời gian/epoch và đỉnh bộ nhớ được đo trong bảng trên. "
               "Ở lần chạy này FP16 không nhanh hơn FP32; MLP nhỏ có thể chịu overhead kernel. "
               "FP16 dùng GradScaler để tránh underflow; BF16 có dải số mũ rộng hơn."
               if "amp-fp16" in by_id else
               "4. Môi trường hiện tại chưa thực hiện thí nghiệm mixed precision. "
               "Chỉ đáng gọi là nhanh hơn khi `time_per_epoch_s` thực đo giảm. "
               "FP16 cần GradScaler để tránh underflow; BF16 có dải số mũ rộng hơn."),
              "5. **Vì sao zeros hỏng; He khác Xavier?** Khởi tạo toàn 0 giữ nơ-ron cùng lớp đối xứng và ReLU(0) có gradient 0; "
              "He dùng phương sai 2/fan_in phù hợp ReLU, Xavier dùng 2/(fan_in+fan_out). "
              "Thí nghiệm thực tế ở đây so He với Xavier.",
              "6. **Loss không giảm sau 2.000 bước: ba kiểm tra đầu tiên?** (i) kiểm tra loss bước 0 và chuẩn hóa để tìm "
              "logit sai thang; (ii) quá khớp 20 mẫu để tìm lỗi nhãn/vòng lặp; "
              "(iii) in gradient từng tham số và chuẩn toàn cục để kiểm tra dòng gradient/lr.", "",
              "## 6. Hạn chế và điều bất ngờ", "",
              ("FP16 chậm hơn FP32 trên T4 và không giảm bộ nhớ đỉnh rõ rệt trong lần đo này; "
               "overhead là giả thuyết, chưa đo tách riêng. " if "amp-fp16" in by_id else ""),
              "Chỉ hai seed baseline nên 2σ còn nhiễu. Chọn LR bằng 5 epoch có thể khác lựa chọn "
              "ở 20 epoch. Mỗi lần thay optimizer đồng thời cần chỉnh LR riêng, vì vậy giải thích bằng "
              "cặp optimizer–LR. Kết luận cơ chế ở trên là giả thuyết phù hợp quan sát, chưa phải "
              "chứng minh nhân quả độc lập. Chưa thử đầy đủ "
              "mọi mức dropout/clipping/init. Cắt gradient với ngưỡng 1,0 cho chênh lệch nhỏ hơn 2σ; "
              "chưa có cặp thử nghiệm LR cao có/không clip. Nếu có thêm thời gian, "
              "lặp các cấu hình tốt với nhiều seed và đếm số bước clip; chưa đo BF16 hoặc nhiễu eval.", "",
              "## 7. Phụ lục", "",
              "File nộp: `code/`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, "
              "`figures/`, `results/`, `REPORT.md`.",
              f"Tổng thời gian huấn luyện ước tính: "
              f"{sum(row['epochs'] * row['time_per_epoch_s'] for row in rows):.0f} giây "
              "theo bảng, chưa gồm chuẩn bị dữ liệu, eval và lưu ảnh.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
