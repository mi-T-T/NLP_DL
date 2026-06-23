import os
import json
import pandas as pd

def audit_vsec_dataset(file_path, output_report_path):
    # Tự động tạo thư mục outputs nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    
    # Mở file để ghi kết quả audit vào thư mục outputs
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("BÁO CÁO KIỂM ĐỊNH CHẤT LƯỢNG DỮ LIỆU (DATA AUDIT REPORT) - VSEC\n")
        f.write("="*60 + "\n\n")
        
        # 1. Kiểm tra sự tồn tại của file nguồn
        if not os.path.exists(file_path):
            f.write(f"[ERROR] Không tìm thấy file dữ liệu gốc tại: {file_path}\n")
            print(f"[ERROR] Thất bại! Không tìm thấy file {file_path}")
            return
        
        # 2. Đọc dữ liệu Parquet
        try:
            df = pd.read_parquet(file_path)
            f.write(f"[INFO] Tải dữ liệu thành công.\n")
            f.write(f"- Tổng số dòng (câu) có trong tập dữ liệu: {len(df)}\n\n")
        except Exception as e:
            f.write(f"[ERROR] Lỗi hệ thống khi đọc file Parquet: {e}\n")
            print(f"[ERROR] Lỗi khi đọc dữ liệu.")
            return

        # 3. Kiểm tra cấu trúc các cột (Schema Audit)
        f.write("--- 1. KIỂM TRA CẤU TRÚC (SCHEMA AUDIT) ---\n")
        expected_fields = [
            'text', 'corrected_text', 'syllable_annotations', 'error_count', 
            'error_positions', 'correction_pairs', 'has_errors'
        ]
        missing_fields = [field for field in expected_fields if field not in df.columns]
        if missing_fields:
            f.write(f"[WARNING] Phát hiện thiếu cột dữ liệu: {missing_fields}\n")
        else:
            f.write("[OK] Cấu trúc các cột đầy đủ và hợp lệ theo đặc tả hệ thống.\n")
        
        f.write("\nKiểu dữ liệu thực tế của từng cột:\n")
        f.write(df.dtypes.to_string() + "\n\n")

        # 4. Kiểm tra Giá trị khuyết (Missing Values)
        f.write("--- 2. KIỂM TRA GIÁ TRỊ KHUYẾT (MISSING VALUES) ---\n")
        null_counts = df.isnull().sum()
        f.write(null_counts.to_string() + "\n")
        if null_counts.sum() == 0:
            f.write("[OK] Tuyệt vời! Không phát hiện giá trị trống (Null/NaN).\n\n")
        else:
            f.write("[WARNING] Cảnh báo: Có giá trị khuyết! Cần kiểm tra lại dữ liệu.\n\n")

        # 5. Phân tích tính toàn vẹn Logic (Consistency Audit)
        f.write("--- 3. KIỂM TRA LOGIC & TÍNH TOÀN VẸN DỮ LIỆU ---\n")
        all_have_errors = df['has_errors'].all()
        f.write(f"- Tất cả các câu đều chứa lỗi (`has_errors = True`)?: {all_have_errors}\n")
        
        logic_errors = 0
        total_errors_counted = 0
        unique_errors = set()
        
        for idx, row in df.iterrows():
            err_count = row['error_count']
            
            # Xử lý an toàn cấu trúc mảng/đối tượng
            try:
                err_pos = row['error_positions'] if isinstance(row['error_positions'], (list, tuple)) else json.loads(row['error_positions'])
                corr_pairs = row['correction_pairs'] if isinstance(row['correction_pairs'], (list, tuple)) else json.loads(row['correction_pairs'])
            except Exception:
                err_pos, corr_pairs = [], []

            total_errors_counted += err_count
            
            for pair in corr_pairs:
                if isinstance(pair, dict) and 'error' in pair:
                    unique_errors.add(pair['error'].lower())

            if len(err_pos) != err_count or len(corr_pairs) != err_count:
                logic_errors += 1

        f.write(f"- Số dòng bị bất nhất cấu trúc (độ dài mảng lỗi != error_count): {logic_errors}\n")
        if logic_errors == 0:
            f.write("  [OK] Tính logic hoàn toàn đồng bộ.\n\n")
        else:
            f.write("  [WARNING] Phát hiện lệch pha đếm số lượng lỗi.\n\n")

        # 6. Thống kê tổng hợp (Statistical Summary)
        f.write("--- 4. SỐ LIỆU THỐNG KÊ CHI TIẾT TẬP DỮ LIỆU ---\n")
        f.write(f"- Tổng số lỗi chính tả (Total Errors): {total_errors_counted}\n")
        f.write(f"- Số lượng từ viết sai độc nhất (Unique Error Types): {len(unique_errors)}\n")
        f.write(f"- Tỷ lệ lỗi trung bình trên một câu: {df['error_count'].mean():.2f}\n")
        f.write(f"- Số lỗi nhiều nhất xuất hiện trong một câu: {df['error_count'].max()}\n\n")
        
        # 7. Trích xuất mẫu ngẫu nhiên (Sanity Check)
        f.write("--- 5. TRÍCH MẪU NGẪU NHIÊN KIỂM TRA (SANITY CHECK) ---\n")
        sample_df = df.sample(min(2, len(df)))
        for idx, row in sample_df.iterrows():
            f.write(f"Câu gốc:       {row['text']}\n")
            f.write(f"Câu đã sửa:    {row['corrected_text']}\n")
            f.write(f"Số lượng lỗi:  {row['error_count']}\n")
            f.write("-" * 50 + "\n")

        f.write("\n" + "="*60 + "\n")
        f.write("HOÀN THÀNH QUÁ TRÌNH KIỂM ĐỊNH\n")
        f.write("="*60 + "\n")

    print(f"[Xong] Quá trình audit hoàn tất! Kết quả đã được xuất thành công vào file: {output_report_path}")

if __name__ == "__main__":
    # Cấu hình đường dẫn đầu vào và đầu ra
    INPUT_DATA = os.path.join("data", "raw", "train.parquet")
    OUTPUT_REPORT = os.path.join("outputs", "data_audit.txt")
    
    audit_vsec_dataset(INPUT_DATA, OUTPUT_REPORT)