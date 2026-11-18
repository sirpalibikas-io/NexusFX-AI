import os
import csv

DATA_DIR = "data"
SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_INR"]
LOOKBACK = 30  # Number of past days used to predict the next day

def create_sequences(data, lookback):
    """Creates sliding window sequences for time-series modeling."""
    x_seq, y_seq = [], []
    for i in range(len(data) - lookback):
        x_seq.append(data[i : i + lookback])
        y_seq.append(data[i + lookback])
    return x_seq, y_seq

def process_and_save_data(symbol):
    filename = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if not os.path.exists(filename):
        print(f"File not found for {symbol}")
        return

    print(f"Preprocessing sequences for {symbol}...")
    
    closes = []
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('close'):
                closes.append(float(row['close']))
                
    if not closes:
        print(f"No valid price data found for {symbol}")
        return
        
    # Min-Max Scaling (Normalization between 0 and 1)
    min_val = min(closes)
    max_val = max(closes)
    
    if max_val - min_val == 0:
        normalized = [0.0 for _ in closes]
    else:
        normalized = [(price - min_val) / (max_val - min_val) for price in closes]
        
    # Generate input sequences (X) and target labels (y)
    X, y = create_sequences(normalized, LOOKBACK)
    
    print(f" -> {symbol}: Generated {len(X)} training samples (Lookback window: {LOOKBACK} days).\n")
    return X, y

if __name__ == "__main__":
    for symbol in SYMBOLS:
        process_and_save_data(symbol)
    print("All preprocessing and sequence generation tasks completed!")