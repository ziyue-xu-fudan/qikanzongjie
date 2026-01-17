import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW
from model import AutismClassifier
import os

# Configuration
DATA_PATH = "../data/dummy_data.csv"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-5

class ClinicalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.loc[idx, 'text']
        label = self.df.loc[idx, 'label']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ClinicalDataset(DATA_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AutismClassifier(MODEL_NAME)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "autism_model.pth")
    print("Model saved to autism_model.pth")

if __name__ == "__main__":
    # Ensure data path is correct relative to execution
    if not os.path.exists(DATA_PATH):
        # Fallback for running from src directory
        DATA_PATH = "data/dummy_data.csv"
        if not os.path.exists(DATA_PATH):
             # Fallback absolute path for demo
             DATA_PATH = "/Users/ziyuexu/Documents/trae_projects/paper1/autism_llm_reproduction/data/dummy_data.csv"
             
    train()
