import os
import time
import requests
import csv
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Check your .env file.")

# Define the currency pairs and output directory
SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/INR"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_forex_data(symbol):
    print(f"Fetching data for {symbol}...")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=1200&apikey={API_KEY}&format=JSON"
    
    response = requests.get(url)
    data = response.json()
    
    if "values" not in data:
        print(f"Error fetching {symbol}: {data}")
        return
    
    values = data["values"]
    
    # Sort chronologically by datetime using standard Python sorting
    values.sort(key=lambda x: x['datetime'])
    
    # Save to CSV using Python's built-in csv module
    filename = os.path.join(DATA_DIR, f"{symbol.replace('/', '_')}_daily.csv")
    
    if values:
        keys = values[0].keys()
        with open(filename, mode='w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(values)
            
        print(f"Saved {symbol} data to {filename}")

if __name__ == "__main__":
    for i, symbol in enumerate(SYMBOLS):
        fetch_forex_data(symbol)
        # Sleep for 8 seconds between requests to respect Twelve Data rate limits (except after the last item)
        if i < len(SYMBOLS) - 1:
            print("Waiting 8 seconds to respect rate limits...")
            time.sleep(8)
            
    print("All data collection tasks completed!")