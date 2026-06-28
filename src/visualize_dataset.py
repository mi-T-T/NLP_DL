import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Nhập bộ tokenizer dùng chung để đếm token đồng bộ với hệ thống
from utils.tokenizer import SimpleVietnameseTokenizer

def analyze_error_positions(noisy_text, clean_text):
    """
    Hàm xác định vị trí tương đối của lỗi trong câu.
    Trả về danh sách các vị trí tỉ lệ phần trăm (từ 0.0 đến 1.0) nơi lỗi xảy ra.
    """
    n_words = str(noisy_text).split()
    c_words = str(clean_text).split()
    
    positions = []
    max_len = min(len(n_words), len(c_words))
    
    if max_len == 0:
        return positions
        
    for i in range(max_len):
        if n_words[i].lower() != c_words[i].lower():
            # Tính vị trí tương đối của từ lỗi trong câu (0.0: Đầu câu, 0.5: Giữa câu, 1.0: Cuối câu)
            relative_pos = i / max_len
            positions.append(relative_pos)
            
    return positions

def main():
    # 1. Đường dẫn file raw parquet (Bạn thay đổi tên file thực tế của bạn tại đây)
    parquet_path = "data/raw_dataset.parquet" 
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(parquet_path):
        # Tạo dữ liệu giả lập định dạng Parquet để test nếu chưa có file thật
        print(f"[INFO] Không tìm thấy {parquet_path}, đang tạo file giả lập để chạy thử...")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        sample_data = pd.DataFrame({
            "noisy_text": ["hom nai troi dep qua", "toi di hoc nhung tre", "vsec nlp DL", "sai chinh ta roi nhe"],
            "clean_text": ["hôm nay trời đẹp quá", "tôi đi học nhưng trễ", "vsec nlp DL", "sai chính tả rồi nhé"]
        })
        sample_data.to_parquet(parquet_path, index=False)

    # 2. Đọc dữ liệu từ file Parquet
    print(f"[PROCESS] Đang đọc dữ liệu từ file Parquet: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["noisy_text", "clean_text"])

    # 3. Trích xuất các đặc trưng thống kê
    print("[PROCESS] Đang tính toán các chỉ số đặc trưng câu và vị trí lỗi...")
    tokenizer = SimpleVietnameseTokenizer()
    
    # Độ dài câu theo ký tự
    df["char_length"] = df["noisy_text"].apply(lambda x: len(str(x)))
    
    # Số lượng từ/token trong câu
    df["word_count"] = df["noisy_text"].apply(lambda x: len(tokenizer.clean_and_split(x)))
    
    # Số lượng lỗi trên mỗi câu (đếm số từ khác nhau giữa câu nhiễu và câu sạch)
    df["error_count"] = df.apply(
        lambda row: sum(1 for n, c in zip(str(row["noisy_text"]).split(), str(row["clean_text"]).split()) if n.lower() != c.lower()) 
                    + abs(len(str(row["noisy_text"]).split()) - len(str(row["clean_text"]).split())), 
        axis=1
    )
    
    # Thu thập toàn bộ vị trí lỗi tương đối
    all_error_positions = []
    for _, row in df.iterrows():
        all_error_positions.extend(analyze_error_positions(row["noisy_text"], row["clean_text"]))

    # 4. KHỞI TẠO VÀ VẼ CÁC BIỂU ĐỒ (Bố cục 2x2)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("BÁO CÁO PHÂN TÍCH DỮ LIỆU THÔ (EDA) - THỐNG KÊ LỖI CHÍNH TẢ", fontsize=18, fontweight='bold', y=0.95)

    # Biểu đồ 1: Phân bố độ dài câu (Ký tự)
    sns.histplot(df["char_length"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("1. Phân bố độ dài câu (Tính theo số Ký tự)", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Số lượng ký tự trong một câu")
    axes[0, 0].set_ylabel("Tần suất (Số câu)")

    # Biểu đồ 2: Phân bố số lượng từ/token trong câu
    sns.histplot(df["word_count"], bins=20, kde=True, ax=axes[0, 1], color="salmon")
    axes[0, 1].set_title("2. Phân bố số lượng Từ / Token trong câu", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Số lượng từ")
    axes[0, 1].set_ylabel("Tần suất (Số câu)")

    # Biểu đồ 3: Tần suất số lượng lỗi xuất hiện trên mỗi câu
    # Sử dụng discrete=True vì số lượng lỗi luôn là số nguyên lẻ (1, 2, 3...)
    sns.histplot(df["error_count"], discrete=True, ax=axes[1, 0], color="teal", alpha=0.8)
    axes[1, 0].set_title("3. Tần suất số lượng từ bị lỗi trên mỗi câu", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Số từ bị lỗi trong một câu")
    axes[1, 0].set_ylabel("Tần suất (Số câu)")
    axes[1, 0].set_xaxis_locator(plt.MaxNLocator(integer=True))

    # Biểu đồ 4: Vị trí xuất hiện của lỗi trong câu
    if all_error_positions:
        sns.kdeplot(all_error_positions, fill=True, bw_adjust=0.5, ax=axes[1, 1], color="purple")
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axes[1, 1].set_xticklabels(["Đầu câu (0%)", "25%", "Giữa câu (50%)", "75%", "Cuối câu (100%)"])
        axes[1, 1].set_title("4. Mật độ vị trí xuất hiện lỗi trong cấu trúc câu", fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel("Vị trí tương đối của từ bị lỗi")
        axes[1, 1].set_ylabel("Mật độ phân bố lỗi")
    else:
        axes[1, 1].text(0.5, 0.5, "Không phát hiện lỗi nào trong tập dữ liệu", ha='center', va='center')
        axes[1, 1].set_title("4. Vị trí xuất hiện lỗi (Không có dữ liệu)")

    # 5. Tối ưu hóa hiển thị và lưu file ảnh kết quả
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    chart_output_path = os.path.join(output_dir, "dataset_eda_report.png")
    plt.savefig(chart_output_path, dpi=200)
    plt.close()
    
    print(f"\n[SUCCESS] Đã khởi tạo và xuất biểu đồ EDA tổng hợp tại: {chart_output_path}")

if __name__ == "__main__":
    main()