import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer

def calculate_over_correction_rate(noisy_list, clean_list, pred_list):
    """
    Tính tỉ lệ Sửa Sai (Over-correction Rate): Mô hình tự ý sửa đổi ở những từ vốn dĩ đã đúng.
    Công thức: (Số từ vốn đúng nhưng bị mô hình thay đổi) / (Tổng số từ vốn đúng)
    """
    total_originally_correct_words = 0
    over_corrected_words = 0
    
    for noisy, clean, pred in zip(noisy_list, clean_list, pred_list):
        n_words = str(noisy).split()
        c_words = str(clean).split()
        p_words = str(pred).split()
        
        # Để so sánh công bằng cấp độ từ, chúng ta duyệt theo độ dài tối thiểu của câu
        min_len = min(len(n_words), len(c_words), len(p_words))
        
        for i in range(min_len):
            # Nếu từ gốc (noisy) và từ chuẩn (clean) giống nhau -> Từ này vốn dĩ ĐÚNG
            if n_words[i].lower() == c_words[i].lower():
                total_originally_correct_words += 1
                # Nếu từ mô hình dự đoán (pred) lại khác từ gốc -> Bị Over-correction
                if p_words[i].lower() != n_words[i].lower():
                    over_corrected_words += 1
                    
    if total_originally_correct_words == 0:
        return 0.0
    return over_corrected_words / total_originally_correct_words

def calculate_correction_f1(noisy_list, clean_list, pred_list):
    """
    Tính F1-Score cho bài toán phát hiện và sửa lỗi (Correction F1)
    - TP: Từ sai được mô hình phát hiện và sửa đúng thành từ chuẩn.
    - FP: Từ đúng bị mô hình sửa (hoặc từ sai sửa thành một từ sai khác).
    - FN: Từ sai nhưng mô hình bỏ sót không sửa.
    """
    tp, fp, fn = 0, 0, 0
    
    for noisy, clean, pred in zip(noisy_list, clean_list, pred_list):
        n_words = str(noisy).split()
        c_words = str(clean).split()
        p_words = str(pred).split()
        
        min_len = min(len(n_words), len(c_words), len(p_words))
        
        for i in range(min_len):
            is_originally_error = (n_words[i].lower() != c_words[i].lower())
            is_modified_by_model = (n_words[i].lower() != p_words[i].lower())
            is_correctly_fixed = (p_words[i].lower() == c_words[i].lower())
            
            if is_originally_error and is_modified_by_model and is_correctly_fixed:
                tp += 1
            elif (not is_originally_error and is_modified_by_model) or (is_originally_error and is_modified_by_model and not is_correctly_fixed):
                fp += 1
            elif is_originally_error and not is_modified_by_model:
                fn += 1
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def evaluate_and_log_to_wandb(run_name, noisy_list, clean_list, pred_list, error_types_list=None):
    """
    Hàm tổng hợp tính toàn bộ Metric, vẽ biểu đồ phân bố và lưu/đẩy trực tiếp lên W&B
    """
    print(f"[METRIC] Đang tính toán các chỉ số kiểm định cho {run_name}...")
    
    # Ép kiểu string an toàn
    noisy_list = [str(x) for x in noisy_list]
    clean_list = [str(x) for x in clean_list]
    pred_list = [str(x) for x in pred_list]
    
    # 1. Tính toán các metric cốt lõi bằng jiwer và các hàm tùy biến
    current_wer = wer(clean_list, pred_list)
    current_cer = cer(clean_list, pred_list)
    current_f1 = calculate_correction_f1(noisy_list, clean_list, pred_list)
    current_over_correct = calculate_over_correction_rate(noisy_list, clean_list, pred_list)
    
    metrics_dict = {
        "Word_Error_Rate_WER": current_wer,
        "Character_Error_Rate_CER": current_cer,
        "Correction_F1_Score": current_f1,
        "Over_Correction_Rate": current_over_correct
    }
    
    # In ra màn hình console kết quả nhanh
    print("-" * 40)
    for k, v in metrics_dict.items():
        print(f" * {k}: {v:.4f}")
    print("-" * 40)
    
    # 2. Tạo và lưu biểu đồ so sánh cục bộ vào thư mục outputs/
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    sns.barplot(x=values, y=names, palette="viridis")
    plt.title(f"Báo cáo chất lượng mô hình - {run_name}")
    plt.xlim(0, max(max(values) + 0.1, 1.0))
    for index, value in enumerate(values):
        plt.text(value, index, f" {value:.4f}")
        
    plot_path = os.path.join("outputs", f"{run_name}_metrics_chart.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Đã lưu biểu đồ phân tích tại: {plot_path}")
    
    # 3. Log trực tiếp các đại lượng và ảnh biểu đồ lên dự án Weights & Biases
    if wandb.run is not None:
        wandb.log(metrics_dict)
        wandb.log({"Metrics_Chart_Image": wandb.Image(plot_path)})
        print(f"[WandB] Đã đồng bộ thành công dữ liệu Run lên hệ thống trực tuyến.")
        
    return metrics_dict