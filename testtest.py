# test_fix.py
import yfinance as yf

stock = yf.Ticker("AMZN")
df = stock.history(period="5d")

print("Before processing:")
print("Columns:", list(df.columns))

# Apply fix
df.columns = df.columns.str.lower().str.replace(' ', '')
if 'adjclose' not in df.columns and 'close' in df.columns:
    df['adjclose'] = df['close']
    print("Created 'adjclose' from 'close'")

print("After processing:")
print("Columns:", list(df.columns))
print("adjclose exists:", 'adjclose' in df.columns)