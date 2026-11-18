import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_INR"]
LOOKBACK = 30

class ForexANN(nn.Module):
    def __init__(self, input_dim=LOOKBACK, hidden_dim=64, output_dim=1):
        super(ForexANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.squeeze(-1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class ForexLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(ForexLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_save_models(symbol):
    filename = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(filename):
        print(f"File not found for {symbol}")
        return

    print(f"\n========================================")
    print(f" Training & Validating Models for {symbol}")
    print(f"========================================")
    
    closes = []
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('close'):
                closes.append(float(row['close']))
                
    min_val, max_val = min(closes), max(closes)
    normalized = [(p - min_val) / (max_val - min_val) for p in closes]
    
    # Save scaling metadata for inference
    meta_path = os.path.join(MODEL_DIR, f"{symbol}_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as mf:
        json.dump({"min_val": min_val, "max_val": max_val}, mf)

    # Generate sliding window sequences
    X, y = [], []
    for i in range(len(normalized) - LOOKBACK):
        X.append(normalized[i : i + LOOKBACK])
        y.append(normalized[i + LOOKBACK])
        
    # Chronological Train/Validation Split (80% Train, 20% Validation)
    train_size = int(len(X) * 0.8)
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
    
    epochs = 50
    criterion = nn.MSELoss()
    
    # ==========================
    # 1. Train & Validate ANN
    # ==========================
    print(f"\n-> Training ANN...")
    ann_model = ForexANN()
    optimizer_ann = optim.Adam(ann_model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        ann_model.train()
        optimizer_ann.zero_grad()
        loss = criterion(ann_model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer_ann.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ann_model.eval()
            with torch.no_grad():
                val_loss = criterion(ann_model(X_val_tensor), y_val_tensor)
            print(f"   Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")
    
    torch.save(ann_model.state_dict(), os.path.join(MODEL_DIR, f"{symbol}_ann.pt"))

    # ==========================
    # 2. Train & Validate LSTM
    # ==========================
    print(f"\n-> Training LSTM...")
    lstm_model = ForexLSTM()
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        lstm_model.train()
        optimizer_lstm.zero_grad()
        loss = criterion(lstm_model(X_train_tensor), y_train_tensor)
        loss.backward()
        optimizer_lstm.step()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lstm_model.eval()
            with torch.no_grad():
                val_loss = criterion(lstm_model(X_val_tensor), y_val_tensor)
            print(f"   Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")
        
    torch.save(lstm_model.state_dict(), os.path.join(MODEL_DIR, f"{symbol}_lstm.pt"))
    print(f" Successfully saved ANN, LSTM, and metadata for {symbol}\n")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        train_and_save_models(symbol)
    print("========================================")
    print(" All models successfully trained and saved!")
    print("========================================")