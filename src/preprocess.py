import os
import re
import unicodedata
import pandas as pd
from underthesea import word_tokenize

# --- BỘ CHUẨN HÓA DẤU TIẾNG VIỆT ---
# Giúp chuyển đổi các từ gõ sai vị trí dấu (ví dụ: hoả -> hỏa, hoà -> hòa)
bang_nguyen_am = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]

nguyen_am_to_ids = {}
for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

def chuan_hoa_dau_tu_tieng_viet(word):
    """
    Chuẩn hóa vị trí đặt dấu chuẩn cho từng từ tiếng Việt độc lập
    """
    if not word:
        return word
    word = unicodedata.normalize('NFC', word)
    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char.lower(), (-1, -1))
        if x != -1:
            if y != 0:
                dau_cau = y
                chars[index] = bang_nguyen_am[x][0]
                if char.isupper():
                    chars[index] = chars[index].upper()
            nguyen_am_index.append(index)
    if len(nguyen_am_index) == 0:
        return word
    if len(nguyen_am_index) == 1:
        chars[nguyen_am_index[0]] = bang_nguyen_am[nguyen_am_to_ids[chars[nguyen_am_index[0]].lower()][0]][dau_cau]
    else:
        # Quy tắc đặt dấu kiểu mới (hòa, nhòe,...)
        if len(nguyen_am_index) == 2:
            if nguyen_am_index[1] == len(chars) - 1:
                idx = nguyen_am_index[0]
            else:
                idx = nguyen_am_index[1]
        else:
            idx = nguyen_am_index[1]
        chars[idx] = bang_nguyen_am[nguyen_am_to_ids[chars[idx].lower()][0]][dau_cau]
    
    # Giữ nguyên hoa/thường
    for i, char in enumerate(word):
        if char.isupper():
            chars[i] = chars[i].upper()
            
    return "".join(chars)

def chuan_hoa_van_ban(text):
    """
    Hàm làm sạch văn bản tổng thể
    """
    if not isinstance(text, str):
        return ""
    # 1. Đưa về chuẩn Unicode dựng sẵn (NFC) tránh lỗi rã dấu chữ tiếng Việt
    text = unicodedata.normalize('NFC', text)
    # 2. Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    # 3. Chuẩn hóa dấu từng từ
    words = text.split(" ")
    words = [chuan_hoa_dau_tu_tieng_viet(w) for w in words]
    return " ".join(words)

def preprocess_pipeline(input_path, output_path):
    print("="*60)
    print(f"[START] Bắt đầu tiền xử lý dữ liệu: {input_path}")
    print("="*60)
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Không tìm thấy file gốc tại: {input_path}")
        return
        
    # Read Data
    df = pd.read_parquet(input_path)
    print(f"[INFO] Đã đọc {len(df)} dòng dữ liệu.")
    
    # 1. Chuẩn hóa văn bản (Text Cleaning & Normalization)
    print("[PROCESS] Đang chuẩn hóa văn bản & sửa dấu chuẩn Unicode...")
    df['clean_text'] = df['text'].apply(chuan_hoa_van_ban)
    df['clean_corrected_text'] = df['corrected_text'].apply(chuan_hoa_van_ban)
    
    # 2. Tách từ tiếng Việt (Word Tokenization) cho bài toán NLP
    print("[PROCESS] Đang thực hiện tách từ (Word Tokenization) bằng Underthesea...")
    # Thao tác này giúp mô hình hiểu được cụm từ phức (ví dụ: "môi_trường", "tuyên_truyền")
    df['tokenized_text'] = df['clean_text'].apply(lambda x: word_tokenize(x, format="text"))
    df['tokenized_corrected_text'] = df['clean_corrected_text'].apply(lambda x: word_tokenize(x, format="text"))
    
    # Tạo thư mục lưu trữ nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lưu file sạch
    df.to_parquet(output_path, index=False)
    print(f"[SUCCESS] Đã tiền xử lý xong và lưu dữ liệu sạch tại: {output_path}")
    
    # In ra ví dụ minh họa sau tiền xử lý
    print("\n--- VÍ DỤ MINH HỌA SAU KHI TIỀN XỬ LÝ ---")
    sample = df.iloc[0]
    print(f"Gốc:       {sample['text']}")
    print(f"Sạch:      {sample['clean_text']}")
    print(f"Tách từ:   {sample['tokenized_text']}")
    print("="*60)

if __name__ == "__main__":
    INPUT_PATH = os.path.join("data", "raw", "train.parquet")
    OUTPUT_PATH = os.path.join("data", "processed", "train_clean.parquet")
    
    preprocess_pipeline(INPUT_PATH, OUTPUT_PATH)