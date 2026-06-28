import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from jiwer import wer

def calculate_wer_for_error_type(df_run, target_error_type):
    """
    Lọc các câu chứa loại lỗi mục tiêu và tính toán chỉ số WER riêng cho nhóm đó
    """
    # Lọc các dòng dữ liệu có nhãn lỗi tương ứng trong cột error_types
    sub_df = df_run[df_run["error_types"].dropna().str.contains(target_error_type)]
    
    if len(sub_df) == 0:
        return None
        
    actual_clean = [str(x) for x in sub_df["clean_text"].tolist()]
    predicted = [str(x) for x in sub_df["pred_text"].tolist()]
    
    return wer(actual_clean, predicted)

def main():
    config_path = os.path.join("configs", "run_05_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Khởi tạo WandB cho Run phân tích
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    print("="*60 + f"\n[RUN 05] PIPELINE PHÂN TÍCH ĐÁNH GIÁ SÂU & ĐỐI SÁNH LỖI CHÍNH TẢ\n" + "="*60)

    # Các loại lỗi cần bóc tách thống kê độc lập
    error_categories = ["thieu_dau", "sai_dau", "teencode_viết_tắt", "go_nham_hoac_vung_mien"]
    
    report_data = []

    # Duyệt qua từng file kết quả dự đoán của các Run trước đó
    for run_name, file_path in config["runs_predictions"].items():
        if not os.path.exists(file_path):
            print(f"[WARNING] Bỏ qua {run_name} do không tìm thấy file kết quả tại: {file_path}")
            continue
            
        print(f"[PROCESS] Đang bóc tách hiệu năng của: {run_name}...")
        df_run = pd.read_csv(file_path)
        
        # 1. Tính toán WER tổng thể trên toàn bộ tập Test
        overall_clean = [str(x) for x in df_run["clean_text"].tolist()]
        overall_pred = [str(x) for x in df_run["pred_text"].tolist()]
        overall_wer = wer(overall_clean, overall_pred)
        
        run_metrics = {
            "Model_Name": run_name,
            "Overall_WER": overall_wer
        }
        
        # 2. Tính toán WER phân rã cho từng danh mục lỗi
        for err_cat in error_categories:
            cat_wer = calculate_wer_for_error_type(df_run, err_cat)
            run_metrics[f"WER_{err_cat}"] = cat_wer
            
        report_data.append(run_metrics)

    # Tạo bảng DataFrame tổng hợp kết quả đối sánh
    report_df = pd.DataFrame(report_data)
    
    # Lưu báo cáo tổng hợp dạng .csv phục vụ nghiệm thu tài liệu bài lab
    out_csv_path = config["outputs"]["report_csv_path"]
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    report_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[SUCCESS] Đã xuất bản báo cáo ma trận đối sánh tại: {out_csv_path}")
    print(report_df.to_string(index=False))

    # --- 3. VẼ BIỂU ĐỒ SO SÁNH MA TRẬN LỖI (ERROR BREAKDOWN CHART) ---
    print("\n[PROCESS] Đang khởi tạo biểu đồ trực quan hóa ma trận lỗi...")
    
    # Biến đổi cấu trúc DataFrame từ dạng rộng (wide) sang dạng dài (long) để trực quan hóa bằng Seaborn
    melted_df = report_df.melt(
        id_vars=["Model_Name"], 
        value_vars=[f"WER_{cat}" for cat in error_categories],
        var_name="Error_Category", 
        value_name="Word_Error_Rate_WER"
    )
    # Làm sạch tên hiển thị trên biểu đồ
    melted_df["Error_Category"] = melted_df["Error_Category"].str.replace("WER_", "")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted_df, x="Error_Category", y="Word_Error_Rate_WER", hue="Model_Name", palette="Set2")
    plt.title("Biểu đồ đối sánh chỉ số WER giữa các mô hình theo từng nhóm lỗi cụ thể")
    plt.ylabel("Word Error Rate (WER) - Càng thấp càng tốt")
    plt.xlabel("Danh mục phân loại lỗi")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_chart_path = config["outputs"]["breakdown_chart_path"]
    plt.savefig(out_chart_path)
    plt.close()
    print(f"[SUCCESS] Đã lưu biểu đồ phân tích sâu tại: {out_chart_path}")

    # --- 4. ĐỒNG BỘ BẢNG DỮ LIỆU & BIỂU ĐỒ LÊN WANDB ---
    # Log bảng đối sánh trực tiếp dạng mẫu (Wandb Table) giúp hiển thị trực quan trên Web
    wandb_table = wandb.Table(dataframe=report_df)
    wandb.log({"Overall_Model_Comparison_Table": wandb_table})
    wandb.log({"Error_Breakdown_Comparison_Chart": wandb.Image(out_chart_path)})
    print("[WandB] Đã nạp thành công toàn bộ bảng so sánh đối chứng lên hệ thống trực tuyến.")
    
    wandb.finish()
    print("="*60)

if __name__ == "__main__":
    main()