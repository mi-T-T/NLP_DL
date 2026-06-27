import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset_pipeline(input_csv_path, output_dir):
    print("="*60)
    print("[SPLIT] BẮT ĐẦU CHIA TẬP DỮ LIỆU CHUNG (TRAIN / VAL / TEST)")
    print("="*60)

    if not os.path.exists(input_csv_path):
        print(f"[ERROR] Không tìm thấy file dữ liệu tiền xử lý tại: {input_csv_path}")
        print("[NOTE] Hãy chắc chắn bạn đã chạy thành công `python src/preprocess.py` trước.")
        return

    # 1. Đọc dữ liệu sạch sau tiền xử lý
    df = pd.read_csv(input_csv_path)
    print(f"[INFO] Tổng số mẫu dữ liệu gốc có sẵn: {len(df)}")

    # 2. Thực hiện chia tách dữ liệu (Cố định random_state để tất cả các Run dùng chung)
    # Tỷ lệ: 80% Train, 10% Val, 10% Test
    # Bước 1: Tách 80% cho Train, 20% còn lại cho nhóm tạm thời (Val + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42,  # Khóa seed để không bị thay đổi giữa các lần chạy
        shuffle=True
    )

    # Bước 2: Chia đôi nhóm tạm thời (20%) thành Val (10%) và Test (10%)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        shuffle=True
    )

    # 3. Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Định nghĩa đường dẫn lưu file cụ thể
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    # 4. Xuất dữ liệu ra các file riêng biệt
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    # 5. In báo cáo thống kê số lượng dòng phục vụ tài liệu
    print("\n--- KẾT QUẢ CHIA TẬP DỮ LIỆU THÀNH CÔNG ---")
    print(f"- Tập Huấn luyện (Train Set) : {len(train_df)} mẫu ({len(train_df)/len(df)*100:.1f}%) -> Đã lưu tại: {train_path}")
    print(f"- Tập Kiểm định (Val Set)    : {len(val_df)} mẫu ({len(val_df)/len(df)*100:.1f}%) -> Đã lưu tại: {val_path}")
    print(f"- Tập Kiểm thử (Test Set)    : {len(test_df)} mẫu ({len(test_df)/len(df)*100:.1f}%) -> Đã lưu tại: {test_path}")
    print("="*60)

if __name__ == "__main__":
    # Đường dẫn file dữ liệu đầu vào sau bước tiền xử lý
    INPUT_DATA = os.path.join("data", "processed", "noisy_clean_dataset.csv")
    
    # Thư mục đích dùng để lưu trữ 3 file dữ liệu chung
    OUTPUT_DIRECTORY = os.path.join("data", "split")
    
    split_dataset_pipeline(INPUT_DATA, OUTPUT_DIRECTORY)