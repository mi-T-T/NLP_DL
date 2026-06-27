import os
import pandas as pd
from tqdm import tqdm

def levenshtein_distance(s1, s2):
    """
    Thuật toán tính khoảng cách chỉnh sửa giữa 2 chuỗi ký tự
    """
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

def build_dictionary(df):
    """
    Tạo từ điển các từ đúng (chính xác) từ tập dữ liệu sạch
    """
    print("[INFO] Đang khởi tạo từ điển tiếng Việt chuẩn từ tập dữ liệu sạch...")
    dictionary = set()
    for text in df['clean_text'].dropna():
        words = str(text).split()
        for word in words:
            # Loại bỏ các dấu câu cơ bản dính liền nếu có để làm sạch từ điển
            clean_word = word.strip(".,!?;:()\"'").lower()
            if clean_word:
                dictionary.add(clean_word)
    print(f"[INFO] Tổng số từ độc nhất trong từ điển: {len(dictionary)}")
    return dictionary

def correct_word_edit_distance(word, dictionary, max_distance=2):
    """
    Tìm từ thay thế tốt nhất trong từ điển có khoảng cách Levenshtein nhỏ nhất
    """
    word_lower = word.lower()
    # Nếu từ đã nằm trong từ điển -> Coi như từ đúng, không sửa
    if word_lower in dictionary or word.isdigit():
        return word

    best_candidate = word
    min_dist = max_distance + 1

    # Duyệt từ điển để tìm từ tối ưu
    for candidate in dictionary:
        dist = levenshtein_distance(word_lower, candidate)
        if dist < min_dist:
            min_dist = dist
            best_candidate = candidate
            # Nếu tìm thấy từ có khoảng cách bằng 1, ưu tiên giữ lại gần nhất
            if min_dist == 1: 
                break

    # Giữ nguyên định dạng viết hoa chữ cái đầu nếu từ gốc viết hoa
    if word and word[0].isupper():
        best_candidate = best_candidate.capitalize()

    return best_candidate

def correct_sentence(sentence, dictionary):
    """
    Hàm xử lý sửa lỗi cho cả câu
    """
    words = str(sentence).split()
    corrected_words = [correct_word_edit_distance(w, dictionary) for w in words]
    return " ".join(corrected_words)

def run_01_pipeline(input_path, output_predict_path):
    print("="*60)
    print("[RUN 01] KHỞI CHẠY BASELINE: EDIT DISTANCE RULE-BASED")
    print("="*60)

    if not os.path.exists(input_path):
        print(f"[ERROR] Không tìm thấy file dữ liệu tại: {input_path}")
        print("[NOTE] Bạn cần chạy file `src/preprocess.py` trước để sinh file này.")
        return

    # Đọc dữ liệu đã tiền xử lý
    df = pd.read_csv(input_path)
    
    # Tạo từ điển từ cột sạch
    dictionary = build_dictionary(df)

    # Tiến hành dự đoán/sửa lỗi trên toàn bộ tập dữ liệu nhiễu
    print("[PROCESS] Đang tiến hành sửa lỗi chính tả trên tập dữ liệu nhiễu...")
    predictions = []
    
    # Sử dụng tqdm để vẽ thanh tiến trình trực quan trong bài lab
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pred_text = correct_sentence(row['noisy_text'], dictionary)
        predictions.append(pred_text)

    df['pred_text'] = predictions

    # Tạo thư mục outputs nếu chưa có
    os.makedirs(os.path.dirname(output_predict_path), exist_ok=True)
    
    # Chỉ lưu các cột cần thiết phục vụ bước đánh giá (evaluation) độc lập sau này
    output_df = df[['noisy_text', 'clean_text', 'pred_text', 'error_count', 'error_types']]
    output_df.to_csv(output_predict_path, index=False, encoding='utf-8-sig')
    
    print(f"[SUCCESS] Đã hoàn thành Run 01! Kết quả dự đoán được lưu tại: {output_predict_path}")
    print("="*60)

if __name__ == "__main__":
    INPUT_DATA = os.path.join("data", "processed", "noisy_clean_dataset.csv")
    # File lưu kết quả dự đoán riêng của Run 1
    OUTPUT_PREDICT = os.path.join("outputs", "run_01_predictions.csv")
    
    run_01_pipeline(INPUT_DATA, OUTPUT_PREDICT)