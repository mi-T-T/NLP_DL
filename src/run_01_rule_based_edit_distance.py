import os
import json
import pandas as pd
import wandb
from tqdm import tqdm

# Nhập các hàm dùng chung đã viết ở bước trước
from utils.tokenizer import SimpleVietnameseTokenizer
from utils.metrics import evaluate_and_log_to_wandb

def levenshtein_distance(s1, s2):
    """Thuật toán tính khoảng cách chỉnh sửa giữa 2 chuỗi ký tự"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def correct_sentence_by_edit_distance(sentence, tokenizer, max_distance=2):
    """Sửa lỗi chính tả cả câu bằng từ điển và khoảng cách Levenshtein"""
    words = str(sentence).split()
    corrected_words = []
    
    for word in words:
        clean_w = word.strip(".,!?;:()\"'").lower()
        
        # Nếu từ nằm trong từ điển hoặc là số/ký tự đặc biệt -> Giữ nguyên
        if clean_w in tokenizer.vocab or word.isdigit() or not clean_w:
            corrected_words.append(word)
            continue
            
        # Nếu không có trong từ điển -> Tiến hành tìm từ thay thế có khoảng cách nhỏ nhất
        best_candidate = clean_w
        min_dist = max_distance + 1
        
        for candidate in tokenizer.vocab:
            dist = levenshtein_distance(clean_w, candidate)
            if dist < min_dist:
                min_dist = dist
                best_candidate = candidate
                if min_dist == 1: # Tối ưu hóa: Thấy khoảng cách bằng 1 thì lấy ngay
                    break
                    
        # Giữ nguyên định dạng viết hoa đầu ngữ nếu từ gốc viết hoa
        if word and word[0].isupper():
            best_candidate = best_candidate.capitalize()
            
        corrected_words.append(best_candidate)
        
    return " ".join(corrected_words)

def main():
    # 1. Định nghĩa và nạp file cấu hình từ thư mục configs/
    config_path = os.path.join("configs", "run_01_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # 2. Khởi tạo Weights & Biases (W&B) và truyền toàn bộ cấu hình vào hệ thống tracking
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config # Đẩy trực tiếp dictionary config lên WandB Dashboard
    )
    
    print("="*60)
    print(f"[RUN 01] KHỞI CHẠY MÔ HÌNH: {config['run_name'].upper()}")
    print("="*60)
    
    # 3. Đọc tập dữ liệu (Học từ điển từ tập Train, Đánh giá trên tập Test)
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    
    # 4. Xây dựng từ điển vàng bằng Tokenizer dùng chung từ tập Train sạch
    tokenizer = SimpleVietnameseTokenizer(max_vocab_size=config["hyperparameters"]["max_vocab_size"])
    tokenizer.build_vocab(train_df["clean_text"].dropna().tolist())
    
    # 5. Thực hiện dự đoán (Sửa lỗi) trên tập dữ liệu Test nhiễu
    print(f"[PROCESS] Đang tiến hành sửa lỗi trên tập kiểm thử ({len(test_df)} câu)...")
    pred_sentences = []
    
    max_dist = config["hyperparameters"]["max_distance"]
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred_text = correct_sentence_by_edit_distance(row["noisy_text"], tokenizer, max_distance=max_dist)
        pred_sentences.append(pred_text)
        
    test_df["pred_text"] = pred_sentences
    
    # 6. Tạo thư mục và lưu file kết quả dự đoán riêng cho Run 1
    out_predict_path = config["outputs"]["predict_out_path"]
    os.makedirs(os.path.dirname(out_predict_path), exist_ok=True)
    
    # Giữ lại các trường cốt lõi phục vụ file nghiệm thu và các bước phân tích sâu
    final_output_df = test_df[["noisy_text", "clean_text", "pred_text", "error_count", "error_types"]]
    final_output_df.to_csv(out_predict_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Đã xuất file dự đoán độc lập của Run 01 tại: {out_predict_path}")
    
    # 7. Gọi hàm Đánh giá dùng chung tính toán WER, CER, F1, Over-correction và vẽ biểu đồ tự động đẩy lên W&B
    evaluate_and_log_to_wandb(
        run_name=config["run_name"],
        noisy_list=final_output_df["noisy_text"].tolist(),
        clean_list=final_output_df["clean_text"].tolist(),
        pred_list=final_output_df["pred_text"].tolist()
    )
    
    # Đóng phiên chạy W&B hoàn tất an toàn
    wandb.finish()
    print("="*60)

if __name__ == "__main__":
    main()
    