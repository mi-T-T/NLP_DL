import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
import wandb
from tqdm import tqdm

# Import hàm đánh giá dùng chung của bài lab
from utils.metrics import evaluate_and_log_to_wandb

# --- 1. TRANSFORMATION DATASET ---
class ViT5CorrectionDataset(Dataset):
    def __init__(self, df, tokenizer, max_in_len, max_tgt_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_in_len = max_in_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Thêm tiền tố tác vụ (Prefix) chuẩn để mô hình nhận diện ngữ cảnh sinh chuỗi
        input_text = "Sửa lỗi chính tả: " + str(row["noisy_text"])
        target_text = str(row["clean_text"])
        
        # Mã hóa chuỗi đầu vào nhiễu
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_in_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Mã hóa chuỗi đích chuẩn sạch
        targets = self.tokenizer(
            target_text, 
            max_length=self.max_tgt_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        labels = targets["input_ids"].squeeze(0)
        # Thay thế các token PAD bằng -100 để PyTorch Loss tự bỏ qua không tính toán gradient
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }

# --- 2. PIPELINE HUẤN LUYỆN VÀ SUY LUẬN ---
def main():
    config_path = os.path.join("configs", "run_04_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Khởi tạo tracking trực tuyến lên dự án Weights & Biases
    wandb.init(project=config["project_name"], name=config["run_name"], config=config)
    print("="*60 + f"\n[RUN 04] FINE-TUNING PRE-TRAINED MODEL: {config['hyperparameters']['model_name'].upper()}\n" + "="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Đang sử dụng phần cứng: {device}")

    hparams = config["hyperparameters"]
    
    # Tải bộ Tokenizer gốc của ViT5 phát hành bởi VietAI
    tokenizer = AutoTokenizer.from_pretrained(hparams["model_name"], use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hparams["model_name"]).to(device)

    # Đọc dữ liệu Train/Test chung
    train_df = pd.read_csv(config["data"]["train_path"]).dropna(subset=["noisy_text", "clean_text"])
    test_df = pd.read_csv(config["data"]["test_path"])

    # Tạo DataLoader
    train_dataset = ViT5CorrectionDataset(train_df, tokenizer, hparams["max_input_length"], hparams["max_target_length"])
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)

    # Khởi tạo bộ tối ưu hóa AdamW chuyên dụng cho Transformer mẫu lớn
    optimizer = AdamW(model.parameters(), lr=hparams["learning_rate"])
    total_steps = len(train_loader) * hparams["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # --- VÒNG LẶP HỌC CHUYỂN GIAO (FINE-TUNING LOOP) ---
    print(f"[TRAIN] Bắt đầu quá trình huấn luyện ViT5 trên {hparams['epochs']} Epochs...")
    model.train()
    for epoch in range(hparams["epochs"]):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hparams['epochs']}"):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass sinh loss trực tiếp từ lớp Encoder-Decoder trong Transformers
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            # Kỹ thuật gradient clipping để ổn định trọng số mô hình lớn
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f" -> Epoch {epoch+1} Hoàn tất. Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch+1, "vit5_train_loss": avg_loss})

    # Lưu lại thư mục chứa checkpoint hoàn chỉnh của Transformers
    os.makedirs(config["outputs"]["model_save_dir"], exist_ok=True)
    model.save_pretrained(config["outputs"]["model_save_dir"])
    tokenizer.save_pretrained(config["outputs"]["model_save_dir"])
    print(f"[SUCCESS] Đã đóng gói lưu checkpoint ViT5 tại: {config['outputs']['model_save_dir']}")

    # --- GENERATION & INFERENCE (SUY LUẬN GIẢI MÃ CHUỖI SẠCH) ---
    print(f"[PROCESS] Đang tiến hành suy luận giải mã sửa lỗi trên tập Test ({len(test_df)} câu)...")
    model.eval()
    pred_sentences = []

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            input_text = "Sửa lỗi chính tả: " + str(row["noisy_text"])
            inputs = tokenizer(input_text, max_length=hparams["max_input_length"], truncation=True, return_tensors="pt").to(device)
            
            # Sử dụng giải thuật sinh chữ thông minh (Beam search hoặc Greedy)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=hparams["max_target_length"],
                num_beams=4, # Đảm bảo sinh câu có chất lượng tối ưu nhất
                early_stopping=True
            )
            
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_sentences.append(pred_text)

    test_df["pred_text"] = pred_sentences

    # Lưu file dự đoán độc lập của Run 4 phục vụ so sánh nghiệm thu bài lab
    out_predict_path = config["outputs"]["predict_out_path"]
    final_output_df = test_df[["noisy_text", "clean_text", "pred_text", "error_count", "error_types"]]
    final_output_df.to_csv(out_predict_path, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] Đã lưu file dự đoán Run 04 tại: {out_predict_path}")

    # Gọi hàm tính toán toàn bộ Metrics và đồng bộ Dashboard W&B
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