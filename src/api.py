import os
import json
import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API key securely from environment variables
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not TWELVEDATA_API_KEY:
    raise ValueError("CRITICAL ERROR: TWELVEDATA_API_KEY is missing from your .env file!")

# Initialize FastAPI app
app = FastAPI(title="Forex Prediction API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
LOOKBACK = 30

# Define the ANN Architecture (must match train_models.py)
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

# Define the LSTM Architecture (must match train_models.py)
class ForexLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(ForexLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def fetch_live_data(symbol: str):
    """Fetches recent daily OHLC time-series data from Twelve Data API."""
    formatted_symbol = symbol.replace("_", "/")  # e.g., EUR_USD -> EUR/USD
    url = f"https://api.twelvedata.com/time_series?symbol={formatted_symbol}&interval=1day&outputsize=50&apikey={TWELVEDATA_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "values" not in data:
        raise HTTPException(status_code=400, detail=f"Failed to fetch data from Twelve Data: {data.get('message', 'Unknown error')}")
    
    # Sort chronologically (oldest to newest)
    candles = sorted(data["values"], key=lambda x: x["datetime"])
    
    formatted_candles = [
        {
            "time": c["datetime"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"])
        }
        for c in candles
    ]
    return formatted_candles

@app.get("/")
def serve_terminal():
    """Serves the NexusFX AI Trading Terminal frontend at the root URL."""
    return FileResponse("index.html")

@app.get("/predict/{symbol}")
def predict(symbol: str, model_type: str = "lstm"):
    """
    Predicts the next price for a given symbol using either 'ann' or 'lstm'.
    Example: /predict/EUR_USD?model_type=lstm
    """
    symbol = symbol.upper()
    model_type = model_type.lower()
    
    if symbol not in ["EUR_USD", "GBP_USD", "USD_JPY", "USD_INR"]:
        raise HTTPException(status_code=404, detail="Symbol not supported.")
    if model_type not in ["ann", "lstm"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'ann' or 'lstm'.")

    # 1. Load scaling metadata
    meta_path = os.path.join(MODEL_DIR, f"{symbol}_meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Metadata not found for {symbol}. Train models first.")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    min_val, max_val = meta["min_val"], meta["max_val"]

    # 2. Fetch live OHLC candles
    candles = fetch_live_data(symbol)
    if len(candles) < LOOKBACK:
        raise HTTPException(status_code=400, detail=f"Not enough data returned. Need at least {LOOKBACK} candles.")

    # Take the last 30 days window
    window_candles = candles[-LOOKBACK:]
    window_closes = [c["close"] for c in window_candles]

    # 3. Normalize closing prices using training min/max
    normalized_window = [(p - min_val) / (max_val - min_val) for p in window_closes]

    # Convert to PyTorch Tensor
    if model_type == "ann":
        input_tensor = torch.tensor([normalized_window], dtype=torch.float32)
    else:
        input_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # 4. Load the trained model weights
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{model_type}.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model weights not found at {model_path}.")

    if model_type == "ann":
        model = ForexANN()
    else:
        model = ForexLSTM()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 5. Run Inference
    with torch.no_grad():
        pred_normalized = model(input_tensor).item()

    # 6. Inverse scale prediction back to Forex price
    predicted_price = pred_normalized * (max_val - min_val) + min_val

    return {
        "symbol": symbol,
        "model_used": model_type.upper(),
        "latest_close": window_closes[-1],
        "predicted_next_close": round(predicted_price, 5),
        "lookback_window_used": len(window_candles),
        "history": window_closes,
        "candles": window_candles
    }