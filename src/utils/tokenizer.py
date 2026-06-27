import re
from collections import Counter

class SimpleVietnameseTokenizer:
    """
    Bộ Tokenizer thủ công dùng chung để bóc tách từ, xây dựng từ điển (Vocabulary)
    và chuyển hóa văn bản thành mảng vector số cho các thuật toán học máy / Baseline.
    """
    def __init__(self, max_vocab_size=50000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab = set()

    def clean_and_split(self, text):
        """
        Tách câu thành mảng các từ đơn và loại bỏ dấu câu nhiễu cơ bản
        """
        if not isinstance(text, str):
            return []
        # Chuyển chữ thường và tách từ dựa trên khoảng trắng
        tokens = text.lower().split()
        # Loại bỏ các ký tự dấu câu dính đầu/cuối của từ
        clean_tokens = [t.strip(".,!?;:()\"'") for t in tokens if t.strip(".,!?;:()\"'")]
        return clean_tokens

    def build_vocab(self, clean_texts_list):
        """
        Xây dựng từ điển vàng từ tập văn bản chuẩn sạch
        """
        print("[TOKENIZER] Đang tiến hành tạo dựng từ điển dùng chung từ tập dữ liệu sạch...")
        word_counts = Counter()
        for text in clean_texts_list:
            tokens = self.clean_and_split(text)
            word_counts.update(tokens)
            
        # Lấy các từ phổ biến nhất theo kích thước vocab giới hạn
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        self.vocab = set(self.word2idx.keys()) - {"<PAD>", "<UNK>"}
        print(f"[TOKENIZER] Hoàn thành! Đã nạp {len(self.word2idx)} token vào bộ nhớ từ điển.")

    def text_to_sequence(self, text):
        """
        Chuyển một câu văn bản thành chuỗi các chỉ số số (Vector ID)
        """
        tokens = self.clean_and_split(text)
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens]

    def sequence_to_text(self, sequence):
        """
        Đảo ngược từ mảng số ID thành câu văn bản chữ đọc được
        """
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in sequence])