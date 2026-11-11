import sqlite3
import pandas as pd

# Connect directly to the SQLite database file
conn = sqlite3.connect('history.db')

# Query the trail using standard SQL
query = "SELECT timestamp, filename, label, model_used, confidence FROM predictions ORDER BY timestamp DESC LIMIT 10"
df = pd.read_sql_query(query, conn)

conn.close()

print("--- VISIONGUARD DATABASE TRAIL ---")
print(df.to_string(index=False))
print("----------------------------------")