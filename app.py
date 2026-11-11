import os
import io
import time
import psycopg2
from psycopg2 import pool
from flask import Flask, request, render_template, jsonify, abort, g
from werkzeug.utils import secure_filename
from PIL import Image
from inference import predict_image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# --- ENV VARS ---
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 1))
MODEL_USED = os.environ.get("MODEL_USED", "VisionGuard@champion")

DB_HOST = os.environ.get("POSTGRES_HOST", "db")
DB_NAME = os.environ.get("POSTGRES_DB", "visionguard")
DB_USER = os.environ.get("POSTGRES_USER", "user")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "password")

app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Connect to Postgres with retry logic
def connect_db():
    while True:
        try:
            db_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10, host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS
            )
            print("Connected to Postgres!")
            return db_pool
        except Exception as e:
            print(f"Database connection failed: {e}. Retrying in 2s...")
            time.sleep(2)

db_pool = connect_db()

def get_db():
    return db_pool.getconn()

def put_db(conn):
    db_pool.putconn(conn)

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions
                  (id SERIAL PRIMARY KEY,
                   timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   filename TEXT, label TEXT, confidence REAL, model_used TEXT)''')
    conn.commit()
    cur.close()
    put_db(conn)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('error.html', message="No file part in request."), 400
    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="No selected file."), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return render_template('error.html', message="Wrong file type. Only PNG, JPG, and JPEG are allowed.")

    try:
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.load() 
    except Exception:
        return render_template('error.html', message="File is corrupted or cannot be decoded.")

    probability = predict_image(pil_img)
    label = "AiArtData" if probability >= THRESHOLD else "RealArt"
    confidence = probability if label == "AiArtData" else 1.0 - probability

    conn = get_db()
    cur = conn.cursor()
    cur.execute('''INSERT INTO predictions (filename, label, confidence, model_used) 
                   VALUES (%s, %s, %s, %s)''', (filename, label, confidence, MODEL_USED))
    conn.commit()
    cur.close()
    put_db(conn)

    return render_template('result.html', filename=filename, label=label, confidence=confidence)

@app.route('/history')
def history():
    history_data = []
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT timestamp, filename, label, model_used, confidence FROM predictions ORDER BY timestamp DESC LIMIT 50')
    for row in cur.fetchall():
        history_data.append({
            'timestamp': row[0], 'filename': row[1], 'label': row[2], 
            'model_used': row[3], 'confidence': row[4]
        })
    cur.close()
    put_db(conn)
    return render_template('history.html', history=history_data)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": True, "database": "connected", "debug_mode": app.debug})

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('error.html', message=f"File is too large. Maximum size is {MAX_UPLOAD_MB}MB."), 413

if __name__ == '__main__':
    init_db() 
    app.run(host='0.0.0.0', port=5000, debug=False)