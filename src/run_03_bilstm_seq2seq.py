import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm

# Import hàm đánh giá dùng chung
from utils.metrics import evaluate_and_log_to_wandb

# --- 1. CHARACTER TOKENIZER ---
class CharTokenizer:
    """Bộ tách chữ cấp ký tự để bọc lót cho các lỗi gõ nhầm bàn phím phức tạp"""
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2char = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        
    def fit(self, texts):
        for text in texts:
            for char in str(text):
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char
                    
    def encode(self, text, max_len):
        tokens = [self.char2idx["<SOS>"]] + [self.char2idx.get(c, self.char2idx["<UNK>"]) for c in str(text)] + [self.char2idx["<EOS>"]]
        if len(tokens) < max_len:
            tokens += [self.char2idx["<PAD>"]] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx in [self.char2idx["<PAD>"], self.char2idx["<EOS>"]]:
                break
            if idx != self.char2idx["<SOS>"]:
                chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

# --- 2. PYTORCH DATASET ---
class TextCorrectionDataset(Dataset):
    def __init__(self, df, src_tokenizer, tgt_tokenizer, max_len):
        self.src_data = [src_tokenizer.encode(row["noisy_text"], max_len) for _, row in df.iterrows()]
        self.tgt_data = [tgt_tokenizer.encode(row["clean_text"], max_len) for _, row in df.iterrows()]
        
    def __len__(self):
        return len(self.src_data)
        
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

# --- 3. ARCHITECTURE: SEQ2SEQ WITH BILSTM ---
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        # Gộp trạng thái 2 chiều của BiLSTM thành 1 chiều cho Decoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        return hidden.unsqueeze(0), cell.unsqueeze(0)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
        return outputs

# --- 4. PIPELINE MAIN ---
def main():
    config_path = os.path.join("configs", "run_03_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    print("="*60 + f"\n[RUN 03] KHỞI CHẠY DEEP LEARNING: {config['run_name'].upper()}\n" + "="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Đang sử dụng phần cứng: {device}")

    # Nạp dữ liệu
    train_df = pd.read_csv(config["data"]["train_path"]).dropna(subset=["noisy_text", "clean_text"])
    test_df = pd.read_csv(config["data"]["test_path"])

    # Khởi tạo Tokenizer cấp ký tự chung
    tokenizer = CharTokenizer()
    tokenizer.fit(train_df["noisy_text"])
    tokenizer.fit(train_df["clean_text"])

    # Chuẩn bị DataLoader
    hparams = config["hyperparameters"]
    train_dataset = TextCorrectionDataset(train_df, tokenizer, tokenizer, hparams["max_seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)

    # Khởi tạo Model
    encoder = Encoder(len(tokenizer.char2idx), hparams["embedding_dim"], hparams["hidden_dim"])
    decoder = Decoder(len(tokenizer.char2idx), hparams["embedding_dim"], hparams["hidden_dim"])
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2idx["<PAD>"])

    # Vòng lặp huấn luyện mô hình (Training loop)
    print("[TRAIN] Đang tiến hành huấn luyện mạng BiLSTM Seq2Seq...")
    model.train()
    for epoch in range(hparams["epochs"]):
        epoch_loss = 0
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hparams['epochs']}"):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f" -> Epoch {epoch+1} Complete. Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss})

    # Lưu trọng số mô hình sau khi kết thúc các epoch
    os.makedirs(os.path.dirname(config["outputs"]["model_checkpoint_path"]), exist_ok=True)
    torch.save(model.state_dict(), config["outputs"]["model_checkpoint_path"])
    print(f"[SUCCESS] Đã lưu checkpoint mô hình tại: {config['outputs']['model_checkpoint_path']}")

    # --- INFERENCE (SỬA LỖI TRÊN TẬP KIỂM THỬ) ---
    print(f"[PROCESS] Đang tiến hành suy luận sửa lỗi trên tập Test ({len(test_df)} câu)...")
    model.eval()
    pred_sentences = []

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            src_tokens = tokenizer.encode(row["noisy_text"], hparams["max_seq_len"])
            src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
            
            # Tạo tensor đích rỗng mồi cho giải mã sinh từ tự động (Teacher forcing = 0)
            trg_placeholder = torch.zeros(1, hparams["max_seq_len"], dtype=torch.long).to(device)
            trg_placeholder[0, 0] = tokenizer.char2idx["<SOS>"]
            
            output = model(src_tensor, trg_placeholder, teacher_forcing_ratio=0.0)
            top_predictions = output.squeeze(0).argmax(dim=-1).cpu().numpy()
            
            pred_text = tokenizer.decode(top_predictions)
            pred_sentences.append(pred_text if pred_text.strip() else row["noisy_text"]) # Giữ lại câu gốc nếu decode bị trống

    test_df["pred_text"] = pred_sentences

    # Lưu file dự đoán độc lập của Run 3
    out_predict_path = config["outputs"]["predict_out_path"]
    final_output_df = test_df[["noisy_text", "clean_text", "pred_text", "error_count", "error_types"]]
    final_output_df.to_csv(out_predict_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Đã lưu file dự đoán Run 03 tại: {out_predict_path}")

    # Tính toán Metric và đẩy đồ thị chất lượng lên W&B Dashboard công khai
    evaluate_and_log_to_wandb(
        run_name=config["run_name"],
        noisy_list=final_output_df["noisy_text"].tolist(),
        clean_list=final_output_df["clean_text"].tolist(),
        pred_list=final_output_df["pred_text"].tolist()
    )
    
    wandb.finish()
    print("="*60)

if __name__ == "__main__":
    main()