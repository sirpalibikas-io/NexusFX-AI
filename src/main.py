import os
import csv
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

app = FastAPI(title="Forex Prediction API", version="1.0")

MODEL_DIR = "models"
DATA_DIR = "data"
LOOKBACK = 30

# Define Model Architectures (matching training)
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

class PredictionRequest(BaseModel):
    symbol: str  # e.g., "EUR_USD"
    model_type: str  # "ann" or "lstm"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Forex Exchange Rate Prediction API!"}

@app.post("/predict")
def predict_exchange_rate(req: PredictionRequest):
    symbol = req.symbol.upper()
    model_type = req.model_type.lower()
    
    if symbol not in ["EUR_USD", "GBP_USD", "USD_JPY", "USD_INR"]:
        raise HTTPException(status_code=400, detail="Invalid currency pair symbol.")
    if model_type not in ["ann", "lstm"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'ann' or 'lstm'.")
        
    # 1. Load Metadata (Scaling parameters)
    meta_path = os.path.join(MODEL_DIR, f"{symbol}_meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Metadata file not found for this symbol.")
    
    with open(meta_path, 'r', encoding='utf-8') as mf:
        meta = json.load(mf)
        min_val = meta["min_val"]
        max_val = meta["max_val"]
        
    # 2. Load latest data from CSV to get the last 30-day sequence
    csv_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Data CSV file not found for this symbol.")
        
    closes = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('close'):
                closes.append(float(row['close']))
                
    if len(closes) < LOOKBACK:
        raise HTTPException(status_code=400, detail="Not enough historical data points.")
        
    # Take the last 30 days
    recent_closes = closes[-LOOKBACK:]
    
    # Normalize using stored min/max
    if max_val - min_val == 0:
        normalized = [0.0 for _ in recent_closes]
    else:
        normalized = [(p - min_val) / (max_val - min_val) for p in recent_closes]
        
    # Prepare tensor input shape: [1, 30, 1] for LSTM or [1, 30] for ANN
    input_tensor = torch.tensor([normalized], dtype=torch.float32).unsqueeze(-1)
    
    # 3. Load the appropriate model weights
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{model_type}.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Trained model weights file not found.")
        
    if model_type == "ann":
        model = ForexANN()
    else:
        model = ForexLSTM()
        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 4. Predict
    with torch.no_grad():
        pred_normalized = model(input_tensor).item()
        
    # Un-normalize prediction back to actual currency price scale
    predicted_price = pred_normalized * (max_val - min_val) + min_val
    
    return {
        "symbol": symbol.replace("_", "/"),
        "model_used": model_type.upper(),
        "last_known_price": closes[-1],
        "predicted_next_close": round(predicted_price, 5)
    }