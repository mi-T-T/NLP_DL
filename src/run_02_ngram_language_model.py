import os
import json
import pandas as pd
import wandb
from collections import Counter, defaultdict
from tqdm import tqdm

# Khai báo các hàm dùng chung đã viết từ trước
from utils.tokenizer import SimpleVietnameseTokenizer
from utils.metrics import evaluate_and_log_to_wandb
from run_01_rule_based_edit_distance import levenshtein_distance

class BigramLanguageModel:
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Tham số làm mượt để tránh xác suất bằng 0
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab_size = 0

    def train(self, sentences_list, tokenizer):
        """Huấn luyện mô hình đếm tần suất N-gram từ văn bản sạch"""
        print("[NGRAM] Đang huấn luyện mô hình ngôn ngữ Bigram trên tập dữ liệu sạch...")
        for sentence in sentences_list:
            tokens = tokenizer.clean_and_split(sentence)
            if not tokens:
                continue
            
            # Thêm token bắt đầu câu <s> để học ngữ cảnh từ đầu câu
            tokens = ["<s>"] + tokens
            
            # Cập nhật Unigram
            self.unigram_counts.update(tokens)
            
            # Cập nhật Bigram
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i+1]
                self.bigram_counts[w1][w2] += 1
                
        self.vocab_size = len(tokenizer.word2idx) + 1 # Cộng 1 cho token <s>

    def get_bigram_probability(self, prev_word, current_word):
        """Tính xác suất P(current_word | prev_word) sử dụng Add-alpha Smoothing"""
        w1 = prev_word.lower() if prev_word else "<s>"
        w2 = current_word.lower()
        
        c_w1 = self.unigram_counts[w1]
        c_w1_w2 = self.bigram_counts[w1][w2]
        
        # Áp dụng công thức làm mượt Laplace mở rộng
        prob = (c_w1_w2 + self.alpha) / (c_w1 + self.alpha * self.vocab_size)
        return prob

def correct_sentence_by_ngram(sentence, tokenizer, ngram_model, max_distance=2):
    """Sửa lỗi cả câu kết hợp khoảng cách chỉnh sửa và mô hình ngôn ngữ Bigram"""
    words = str(sentence).split()
    corrected_words = []
    
    for idx, word in enumerate(words):
        clean_w = word.strip(".,!?;:()\"'").lower()
        prev_w = corrected_words[idx-1] if idx > 0 else "<s>"
        
        # Nếu từ hợp lệ trong từ điển, là số hoặc rỗng -> Giữ nguyên không sửa
        if clean_w in tokenizer.vocab or word.isdigit() or not clean_w:
            corrected_words.append(word)
            continue
            
        # Tìm tất cả ứng viên có khoảng cách Levenshtein <= max_distance
        candidates = []
        min_dist = max_distance + 1
        
        for candidate in tokenizer.vocab:
            dist = levenshtein_distance(clean_w, candidate)
            if dist < min_dist:
                min_dist = dist
                candidates = [candidate]
            elif dist == min_dist and dist <= max_distance:
                candidates.append(candidate)
                
        # Nếu không tìm thấy ứng viên nào phù hợp -> Giữ nguyên từ gốc
        if not candidates:
            corrected_words.append(word)
            continue
            
        # Sử dụng N-gram Model để chọn ứng viên có xác suất ngữ cảnh cao nhất
        best_candidate = candidates[0]
        max_prob = -1.0
        
        for candidate in candidates:
            prob = ngram_model.get_bigram_probability(prev_w, candidate)
            if prob > max_prob:
                max_prob = prob
                best_candidate = candidate
                
        # Khôi phục định dạng viết hoa nếu cần
        if word and word[0].isupper():
            best_candidate = best_candidate.capitalize()
            
        corrected_words.append(best_candidate)
        
    return " ".join(corrected_words)

def main():
    # 1. Nạp file cấu hình Run 2
    config_path = os.path.join("configs", "run_02_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    # 2. Khởi tạo kết nối Weights & Biases (W&B)
    wandb.init(
        project=config["project_name"],
        name=config["run_name"],
        config=config
    )
    
    print("="*60)
    print(f"[RUN 02] KHỞI CHẠY MÔ HÌNH: {config['run_name'].upper()}")
    print("="*60)
    
    # 3. Đọc dữ liệu Train/Test từ thư mục split chung
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    
    # 4. Xây dựng từ điển vàng bằng Tokenizer dùng chung
    tokenizer = SimpleVietnameseTokenizer(max_vocab_size=config["hyperparameters"]["max_vocab_size"])
    tokenizer.build_vocab(train_df["clean_text"].dropna().tolist())
    
    # 5. Huấn luyện Mô hình ngôn ngữ Bigram từ văn bản Train sạch
    ngram_model = BigramLanguageModel(alpha=config["hyperparameters"]["smoothing_alpha"])
    ngram_model.train(train_df["clean_text"].dropna().tolist(), tokenizer)
    
    # 6. Sửa lỗi trên tập Test nhiễu bằng sự kết hợp Edit Distance + N-gram
    print(f"[PROCESS] Đang sửa lỗi ngữ cảnh trên tập kiểm thử ({len(test_df)} câu)...")
    pred_sentences = []
    max_dist = config["hyperparameters"]["max_distance"]
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred_text = correct_sentence_by_ngram(row["noisy_text"], tokenizer, ngram_model, max_distance=max_dist)
        pred_sentences.append(pred_text)
        
    test_df["pred_text"] = pred_sentences
    
    # 7. Lưu file dự đoán riêng biệt của Run 2
    out_predict_path = config["outputs"]["predict_out_path"]
    os.makedirs(os.path.dirname(out_predict_path), exist_ok=True)
    
    final_output_df = test_df[["noisy_text", "clean_text", "pred_text", "error_count", "error_types"]]
    final_output_df.to_csv(out_predict_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Đã xuất file dự đoán độc lập của Run 02 tại: {out_predict_path}")
    
    # 8. Tính Metrics và tự động vẽ biểu đồ, log trực tiếp lên Dashboard W&B
    evaluate_and_log_to_wandb(
        run_name=config["run_name"],
        noisy_list=final_output_df["noisy_text"].tolist(),
        clean_list=final_output_df["clean_text"].tolist(),
        pred_list=final_output_df["pred_text"].tolist()
    )
    
    # Đóng phiên kết nối WandB an toàn
    wandb.finish()
    print("="*60)

if __name__ == "__main__":
    main()