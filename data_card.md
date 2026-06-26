#DataCard: VSEC (Vietnamese Spell Correction Dataset)
## 1. Dataset Summary
- Tên tập dữ liệu: Vietnamese Spell Correction (VSEC)

- Ngôn ngữ: Tiếng Việt (Vietnamese)

- Kích thước: 9.341 câu hoàn chỉnh.

- Tổng số lỗi chính tả: 11.202 lỗi do con người tạo ra (human-made misspellings).

- Số lượng loại lỗi độc nhất: 5.211 loại lỗi (Unique error types).

- Tỷ lệ lỗi trung bình: 1.20 lỗi/câu.

- Định dạng lưu trữ gốc: JSONL / Parquet (với các annotation chi tiết ở cấp độ âm tiết - syllable-level).

- Độ bao phủ lỗi: 100% (Tất cả các câu trong tập dữ liệu đều chứa ít nhất một lỗi chính tả).
## 2. Motivation and Intended Use
- Động lực phát triển: Giải quyết sự thiếu hụt các bộ dữ liệu công khai, quy mô lớn và có chất lượng gắn nhãn cao dành riêng cho bài toán phát hiện và sửa lỗi chính tả tiếng Việt. Các bộ dữ liệu trước đây thường quy mô nhỏ hoặc sử dụng thuật toán tự động tạo lỗi (heuristic noise) không phản ánh đúng thực tế hành vi của người dùng.

- Mục đích sử dụng: Được thiết kế làm tập dữ liệu chuẩn để huấn luyện, tối ưu hóa và đánh giá các hệ thống NLP sửa lỗi chính tả tiếng Việt (Vietnamese Spell Correction Systems) sử dụng các kiến trúc như Sequence Tagging, Seq2Seq (Transformer, BARTphoto, PhoBERT).
## 3. Dataset Sources
- Nguồn dữ liệu gốc: Các văn bản tiếng Việt đa dạng ngữ cảnh thu thập từ internet bao gồm: bài báo trực tuyến, bài đăng trên mạng xã hội, diễn đàn, và văn bản hành chính công cộng.

- Bản quyền gốc: Dữ liệu được thu thập công khai từ các nguồn mở và được cộng đồng nghiên cứu xử lý lại.
## 4. Dataset Composition
Mỗi ví dụ (bản ghi) trong tập dữ liệu bao gồm các trường thông tin sau:

- text (string): Câu gốc chứa các lỗi chính tả cần sửa.

- corrected_text (string): Câu đích sau khi đã được sửa đổi hoàn chỉnh về mặt chính tả và ngữ nghĩa.

- syllable_annotations (list): Mảng chứa thông tin chi tiết từng âm tiết 
    - syllable (string): Văn bản của âm tiết
    - is_correct (boolean): Âm tiết có được viết đúng chính tả hay không
    - corrections (list): Danh sách các gợi ý sửa lỗi cho các âm tiết không chính xác
    - position (int): Vị trí của âm tiết trong câu (chỉ số bắt đầu từ 0)

- error_count (int): Tổng số lỗi xuất hiện trong câu.

- error_positions (list): Danh sách các chỉ số vị trí (index) của âm tiết bị lỗi.

- correction_pairs (list): Danh sách các cặp đối sánh thực tế dạng {"error": "...", "correction": "...", "position": ...}.

- has_errors (boolean): Ghi nhận trạng thái có lỗi (luôn là true trong tập train này).
## 5. Data Collection Process
- Thu thập văn bản nền: Hệ thống thu thập tự động văn bản thô từ các nền tảng số hóa tiếng Việt.

- Cơ chế tạo lỗi thực tế: Thay vì tạo nhiễu ngẫu nhiên bằng máy tính, bộ dữ liệu tập trung thu thập hoặc mô phỏng trực tiếp từ các lỗi thực tế của con người:

    - Mistyped errors: Lỗi xảy ra do quá trình gõ phím nhanh (bấm nhầm phím liền kề trên bàn phím Telex/VNI, thiếu ký tự, thừa ký tự).

    - Misspelled errors: Lỗi do phát âm vùng miền (lẫn lộn l/n, ch/tr, s/x, d/v/gi) hoặc do các từ ngữ có cấu trúc âm tiết phức tạp, dễ gây nhầm lẫn ngữ nghĩa.
## 6. Annotation Process
- Phương pháp gắn nhãn: Quá trình gắn nhãn và sửa lỗi được thực hiện hoàn toàn bởi con người nhằm đảm bảo tính tự nhiên và chính xác của câu sửa.

- Chi tiết cấp độ gắn nhãn: Thực hiện chia tách đến cấp độ âm tiết (Syllable-level). Người gắn nhãn tiến hành định vị chính xác vị trí lỗi, phân loại lỗi và đưa ra từ thay thế tối ưu nhất dựa trên ngữ cảnh toàn câu.
## 7. Preprocessing
- Loại bỏ ký tự nhiễu (biểu tượng cảm xúc, xuống dòng)
- Chuẩn hóa chữ hoa và chữ thường
- Chuẩn hóa ký hiệu bằng cách sử dụng phương pháp ánh xạ kiểu đánh máy telex
- Tách âm tiết bằng thuật toán phân đoạn từ
## 8. Train/Validation/Test Split
- Tập huấn luyện (Train set): Chiếm 80% dữ liệu gốc (Phục vụ cập nhật trọng số mô hình).

- Tập kiểm định (Validation set): Chiếm 10% dữ liệu (Phục vụ tuning siêu tham số và tránh Overfitting).

- Tập kiểm thử (Test set): Chiếm 10% dữ liệu (Được giữ độc lập hoàn toàn để đánh giá khách quan các chỉ số BLEU, Character Error Rate - CER, Word Error Rate - WER).
## 9. Label Distribution
- Phân bố số lượng lỗi: Phân bố tập trung mạnh nhất ở các câu có từ 1 đến 2 lỗi chính tả (trung bình 1.20 lỗi/câu), phản ánh đúng hành vi viết văn bản thông thường (ít khi một câu ngắn bị sai hàng loạt từ).

- Phân bố loại lỗi: Tỷ lệ lỗi gõ phím chiếm ưu thế ở các dữ liệu mạng xã hội; trong khi lỗi phát âm vùng miền và chính tả từ phức tạp chiếm tỷ trọng cao hơn ở các dữ liệu dạng bài luận hoặc tin tức.
## 10. Data Quality Checks
- Kiểm tra tính nhất quán (Logic Check): Đã chạy script kiểm tra tự động đảm bảo error_count luôn khớp với số lượng phần tử trong error_positions và correction_pairs.

- Kiểm tra giá trị khuyết: Xác định không có dòng dữ liệu nào bị trống (Null/NaN) ở các trường cốt lõi (text, corrected_text).

- Độ chính xác nhãn: Được kiểm định chéo (Cross-verification) giữa các thành viên gắn nhãn để giảm thiểu sai sót chủ quan.
## 11. Ethical Considerations
- Bảo vệ quyền riêng tư: Toàn bộ dữ liệu đã được ẩn danh hóa. Các thông tin định danh cá nhân nhạy cảm như: tên riêng, số điện thoại, số tài khoản, địa chỉ cụ thể... xuất hiện trong văn bản gốc đều đã bị loại bỏ hoặc thay thế bằng các thực thể chung chung.

- Nội dung lành mạnh: Các văn bản chứa nội dung thù địch, độc hại hoặc vi phạm pháp luật nghiêm trọng đã bị lọc bỏ trong quá trình tiền xử lý.
## 12. Biases and Limitations
- Thiên vị vùng miền: Mặc dù bao gồm cả lỗi phát âm vùng miền, tỷ lệ phân bố lỗi có thể nghiêng dịch về một số lỗi phổ biến của một miền cụ thể (ví dụ: lỗi phụ âm đầu miền Bắc hoặc lỗi hỏi/ngã miền Nam) tùy thuộc vào tỷ lệ nguồn văn bản thu thập được.

- Giới hạn ngữ cảnh ngắn: Dữ liệu được xử lý cô lập theo từng câu riêng lẻ. Do đó, mô hình huấn luyện dựa trên bộ dữ liệu này có thể gặp khó khăn với các lỗi chính tả đòi hỏi phải hiểu ngữ cảnh của cả một đoạn văn dài.
## 13. License and Access
- Bản quyền sử dụng: Cung cấp công khai dưới các giấy phép mã nguồn mở phục vụ mục đích học tập và nghiên cứu phi thương mại.

- Phương thức tiếp cận: Truy cập thông qua kho lưu trữ dữ liệu khoa học công khai (Hugging Face Datasets).
## 14. Recommended Uses
- Phù hợp nhất để xây dựng các mô hình:

    - Phát hiện lỗi chính tả (Spell Detection / Sequence Tagging).

    - Sửa lỗi chính tả tự động (Spell Correction / GEC - Grammatical Error Correction).

    - Tiền xử lý dữ liệu đầu vào cho các hệ thống chatbot, trợ lý ảo (Voice-to-Text post-processing) và công cụ tìm kiếm tiếng Việt.
## 15. Prohibited or Risky Uses
- Nghiêm cấm thương mại hóa trái phép: Không sử dụng bộ dữ liệu trực tiếp vào các sản phẩm thương mại đóng gói thu tiền khi chưa được sự cho phép từ các tác giả/đơn vị chủ quản bộ dữ liệu.

- Rủi ro tạo văn bản rác: Không sử dụng cặp dữ liệu này theo chiều ngược lại (học cách sinh ra lỗi chính tả) với mục đích cố ý tạo ra các văn bản nhiễu, spam, hoặc làm suy giảm chất lượng hiển thị nội dung trên các nền tảng mạng Internet.