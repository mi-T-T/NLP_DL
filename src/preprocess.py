import os
import re
import unicodedata
import pandas as pd
import numpy as np
import ast  # Thêm thư viện này để parse chuỗi dạng python dict (nháy đơn) an toàn

def phan_loai_loi_tu_dong(error_word, correct_word):
    """
    Phân loại loại lỗi dựa trên cặp từ lỗi và từ đúng.
    """
    if not isinstance(error_word, str) or not isinstance(correct_word, str):
        return "khac"
    
    e_w = error_word.lower().strip()
    c_w = correct_word.lower().strip()
    
    if e_w == c_w:
        return "khong_loi"
        
    # 1. Loại bỏ dấu tiếng Việt để kiểm tra lỗi dấu
    def loai_bo_dau(text):
        text = unicodedata.normalize('NFD', text)
        text = re.sub(r'[\u0300-\u036f]', '', text)
        return text.replace('đ', 'd')

    e_no_diacritic = loai_bo_dau(e_w)
    c_no_diacritic = loai_bo_dau(c_w)
    
    if e_no_diacritic == c_no_diacritic:
        if e_w == e_no_diacritic:
            return "thieu_dau"
        return "sai_dau"
        
    # 2. Kiểm tra teencode hoặc viết tắt đơn giản
    if e_w in ['dc', 'khg', 'bít', 'đc', 'mún', 'j', 'ko', 'k']:
        return "teencode_viết_tắt"

    # 3. Các trường hợp còn lại (bao gồm gõ thiếu ký tự như tà -> tài)
    return "go_nham_hoac_vung_mien"

def chuan_hoa_unicode_nfc(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_parse_pairs(pairs_data):
    """
    Hàm ép kiểu và bóc tách dữ liệu cặp lỗi-sửa một cách an toàn tuyệt đối.
    Giải quyết vấn đề lỗi định dạng chuỗi/mảng của pandas khi đọc từ parquet.
    """
    if pairs_data is None:
        return []
        
    # Nếu bản chất đã là list/ndarray sẵn
    if isinstance(pairs_data, (list, np.ndarray, tuple)):
        # Chuyển về list thuần python để duyệt
        return list(pairs_data)
        
    # Nếu bị lưu thành chuỗi kí tự (string) đại diện cho list
    if isinstance(pairs_data, str):
        pairs_data = pairs_data.strip()
        if not pairs_data or pairs_data in ['[]', 'None', 'nan']:
            return []
        try:
            # Thử nghiệm dùng ast.literal_eval trước vì nó chấp nhận cả nháy đơn lẫn nháy kép
            return ast.literal_eval(pairs_data)
        except:
            try:
                # Dự phòng bằng json.loads nếu ast thất bại
                import json
                return json.loads(pairs_data)
            except:
                return []
                
    return []

def preprocess_and_build_dataset(input_parquet_path, output_csv_path):
    print("="*60)
    print(f"[BƯỚC 1] TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG NÂNG CAO (ĐÃ KHẮC PHỤC LỖI PARSE)")
    print("="*60)
    
    if not os.path.exists(input_parquet_path):
        print(f"[ERROR] Không tìm thấy file gốc tại: {input_parquet_path}")
        return
        
    # Đọc dữ liệu
    df = pd.read_parquet(input_parquet_path)
    print(f"[INFO] Đã tải thành công {len(df)} câu từ file Parquet.")
    
    # 1. Chuẩn hóa Unicode NFC
    print("[PROCESS] Thao tác 1: Chuẩn hóa Unicode dựng sẵn (NFC)...")
    df['text'] = df['text'].apply(chuan_hoa_unicode_nfc)
    df['corrected_text'] = df['corrected_text'].apply(chuan_hoa_unicode_nfc)
    
    # 2. Phân loại lỗi
    print("[PROCESS] Thao tác 2: Phân loại nhãn lỗi (Thiếu dấu, sai dấu, gõ nhầm...)...")
    error_types_per_sentence = []
    
    for idx, row in df.iterrows():
        # Gọi hàm bóc tách an toàn mới thiết kế
        pairs = safe_parse_pairs(row['correction_pairs'])
                
        sentence_errors = []
        if isinstance(pairs, list):
            for pair in pairs:
                # Đảm bảo phần tử bóc ra phải là một dictionary
                if isinstance(pair, dict) and 'error' in pair and 'correction' in pair:
                    err_type = phan_loai_loi_tu_dong(pair['error'], pair['correction'])
                    # Không đưa nhãn khong_loi vào danh sách nhãn lỗi của câu
                    if err_type != "khong_loi":
                        sentence_errors.append(err_type)
                    
        # Nếu danh sách trống nhưng has_errors = True, gán nhãn dự phòng
        if len(sentence_errors) == 0 and row['has_errors']:
            sentence_errors.append("go_nham_hoac_vung_mien")
            
        # Lưu chuỗi nhãn phân cách bởi dấu phẩy
        error_types_per_sentence.append(",".join(set(sentence_errors)))
        
    df['error_types'] = error_types_per_sentence
    
    # 3. Chuẩn bị các cột dữ liệu xuất file
    final_df = df[[
        'text', 
        'corrected_text', 
        'error_count', 
        'error_types', 
        'has_errors'
    ]].rename(columns={
        'text': 'noisy_text',
        'corrected_text': 'clean_text'
    })
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Xuất file CSV nghiệm thu
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"[SUCCESS] Đã tạo thành công file cấu trúc: {output_csv_path}")
    
    # Thống kê kết quả thực tế sau khi sửa lỗi parse
    print("\n--- THỐNG KÊ SỐ LƯỢNG CÁC LOẠI LỖI ĐÃ BÓC TÁCH THÀNH CÔNG ---")
    all_types = []
    for t_str in final_df['error_types'].dropna():
        if t_str:
            all_types.extend(t_str.split(','))
    if all_types:
        print(pd.Series(all_types).value_counts())
    else:
        print("[NOTE] Không tìm thấy nhãn lỗi nào.")
    print("="*60)

if __name__ == "__main__":
    INPUT_PATH = os.path.join("data", "raw", "train.parquet")
    OUTPUT_PATH = os.path.join("data", "processed", "noisy_clean_dataset.csv") 
    
    preprocess_and_build_dataset(INPUT_PATH, OUTPUT_PATH)