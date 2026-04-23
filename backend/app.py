"""
Insurance Cost Predictor - Flask Backend
Full-stack with SQLite, JWT auth, XGBoost pipeline
"""
import os
import csv
import json
import sqlite3
import hashlib
import hmac
import secrets
import datetime
import io
from functools import wraps
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
import pandas as pd
import numpy as np
import joblib

# ─── App Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(FRONTEND_DIR, 'static')
)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# ─── Paths ────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join(BASE_DIR, 'insurance_app.db')
MODEL_PATH    = os.path.join(BASE_DIR, 'insurance_model.pkl')
ORIG_CSV      = os.path.join(BASE_DIR, 'insurance.csv')
NEW_INPUTS_CSV= os.path.join(BASE_DIR, 'new_inputs.csv')

# ─── Load Model ───────────────────────────────────────────────────────────────
print("Loading model...")
pipeline = joblib.load(MODEL_PATH)
print("Model loaded ✓")

# ─── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER,
            sex TEXT,
            bmi REAL,
            children INTEGER,
            smoker TEXT,
            region TEXT,
            predicted_charge REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized ✓")

init_db()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 200000)
    return h.hex(), salt

def verify_password(password, password_hash, salt):
    h, _ = hash_password(password, salt)
    return hmac.compare_digest(h, password_hash)

def create_session_token(user_id):
    token = secrets.token_hex(32)
    expires = (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).isoformat()
    conn = get_db()
    conn.execute('INSERT INTO sessions (user_id, token, expires_at) VALUES (?,?,?)',
                 (user_id, token, expires))
    conn.commit()
    conn.close()
    return token

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        conn = get_db()
        row = conn.execute(
            'SELECT s.user_id, s.expires_at, u.email FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?',
            (token,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Invalid token'}), 401
        if datetime.datetime.fromisoformat(row['expires_at']) < datetime.datetime.utcnow():
            return jsonify({'error': 'Token expired'}), 401
        request.user_id = row['user_id']
        request.user_email = row['email']
        return f(*args, **kwargs)
    return decorated

# ─── Model Helpers ────────────────────────────────────────────────────────────
def retrain_model():
    """Retrain on original + all new collected data for better predictions."""
    global pipeline
    try:
        orig_df = pd.read_csv(ORIG_CSV)
        if os.path.exists(NEW_INPUTS_CSV):
            new_df = pd.read_csv(NEW_INPUTS_CSV)
            # new_inputs has no 'charges' col — skip those for retraining labels
            # Only retrain with rows that have charge labels (i.e., original data + any labelled additions)
            combined = orig_df  # for now, original + original is fine
        else:
            combined = orig_df

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split

        X = combined.drop(columns='charges')
        Y = combined['charges']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), ['age', 'bmi', 'children']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
        ])
        new_pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                 max_depth=5, subsample=0.8, random_state=42))
        ])
        new_pipeline.fit(X_train, Y_train)
        joblib.dump(new_pipeline, MODEL_PATH)
        pipeline = new_pipeline
        score = new_pipeline.score(X_test, Y_test)
        print(f"Model retrained. Test R²: {score:.4f}")
        return score
    except Exception as e:
        print(f"Retrain error: {e}")
        return None

# ─── Routes: Pages ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ─── Routes: Auth ─────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    pw_hash, salt = hash_password(password)
    created_at = datetime.datetime.utcnow().isoformat()

    try:
        conn = get_db()
        conn.execute('INSERT INTO users (email, password_hash, salt, created_at) VALUES (?,?,?,?)',
                     (email, pw_hash, salt, created_at))
        conn.commit()
        user_id = conn.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()['id']
        conn.close()
        token = create_session_token(user_id)
        return jsonify({'token': token, 'email': email, 'message': 'Account created successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already registered'}), 409

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    conn.close()

    if not user or not verify_password(password, user['password_hash'], user['salt']):
        return jsonify({'error': 'Invalid email or password'}), 401

    token = create_session_token(user['id'])
    return jsonify({'token': token, 'email': email, 'message': 'Login successful'})

@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    conn = get_db()
    conn.execute('DELETE FROM sessions WHERE token=?', (token,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Logged out'})

# ─── Routes: Predict ──────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
@require_auth
def predict():
    data = request.get_json()
    required = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400

    try:
        input_df = pd.DataFrame([{
            'age':      int(data['age']),
            'sex':      str(data['sex']),
            'bmi':      float(data['bmi']),
            'children': int(data['children']),
            'smoker':   str(data['smoker']),
            'region':   str(data['region'])
        }])
        prediction = pipeline.predict(input_df)[0]

        # Save to DB
        created_at = datetime.datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute(
            'INSERT INTO predictions (user_id,age,sex,bmi,children,smoker,region,predicted_charge,created_at) VALUES (?,?,?,?,?,?,?,?,?)',
            (request.user_id, int(data['age']), data['sex'], float(data['bmi']),
             int(data['children']), data['smoker'], data['region'], float(prediction), created_at)
        )
        conn.commit()
        conn.close()

        # Append to new_inputs.csv (without charges column)
        new_row = {
            'age': int(data['age']), 'sex': data['sex'], 'bmi': float(data['bmi']),
            'children': int(data['children']), 'smoker': data['smoker'], 'region': data['region'],
            'predicted_charge': float(prediction), 'submitted_at': created_at,
            'submitted_by': request.user_email
        }
        file_exists = os.path.exists(NEW_INPUTS_CSV)
        with open(NEW_INPUTS_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)

        return jsonify({'estimated_charges': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Routes: History ──────────────────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
@require_auth
def history():
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 20',
        (request.user_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ─── Routes: Download CSVs ────────────────────────────────────────────────────
@app.route('/api/download/new-inputs', methods=['GET'])
@require_auth
def download_new_inputs():
    """Download the collected new prediction inputs as CSV."""
    if not os.path.exists(NEW_INPUTS_CSV):
        return jsonify({'error': 'No new inputs collected yet'}), 404
    return send_file(
        NEW_INPUTS_CSV,
        mimetype='text/csv',
        as_attachment=True,
        download_name='new_insurance_inputs.csv'
    )

@app.route('/api/download/updated-dataset', methods=['GET'])
@require_auth
def download_updated_dataset():
    """Download original dataset merged with new predicted inputs."""
    orig_df = pd.read_csv(ORIG_CSV)
    if os.path.exists(NEW_INPUTS_CSV):
        new_df = pd.read_csv(NEW_INPUTS_CSV)
        # Align with original columns, use predicted_charge as 'charges'
        append_df = new_df[['age','sex','bmi','children','smoker','region']].copy()
        append_df['charges'] = new_df['predicted_charge']
        combined = pd.concat([orig_df, append_df], ignore_index=True)
    else:
        combined = orig_df

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='updated_insurance_dataset.csv'
    )

# ─── Routes: Retrain ──────────────────────────────────────────────────────────
@app.route('/api/retrain', methods=['POST'])
@require_auth
def retrain():
    score = retrain_model()
    if score is not None:
        return jsonify({'message': 'Model retrained successfully', 'test_r2': round(score, 4)})
    return jsonify({'error': 'Retraining failed'}), 500

# ─── Routes: Stats ────────────────────────────────────────────────────────────
@app.route('/api/stats', methods=['GET'])
@require_auth
def stats():
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) as c FROM predictions WHERE user_id=?', (request.user_id,)).fetchone()['c']
    avg   = conn.execute('SELECT AVG(predicted_charge) as a FROM predictions WHERE user_id=?', (request.user_id,)).fetchone()['a']
    max_c = conn.execute('SELECT MAX(predicted_charge) as m FROM predictions WHERE user_id=?', (request.user_id,)).fetchone()['m']
    min_c = conn.execute('SELECT MIN(predicted_charge) as m FROM predictions WHERE user_id=?', (request.user_id,)).fetchone()['m']
    conn.close()
    total_new = 0
    if os.path.exists(NEW_INPUTS_CSV):
        with open(NEW_INPUTS_CSV) as f:
            total_new = sum(1 for _ in f) - 1
    return jsonify({
        'total_predictions': total,
        'average_charge': round(avg, 2) if avg else 0,
        'max_charge': round(max_c, 2) if max_c else 0,
        'min_charge': round(min_c, 2) if min_c else 0,
        'total_new_inputs': max(total_new, 0)
    })

if __name__ == '__main__':
    print("Starting Insurance Predictor API...")
    app.run(debug=True, port=5000, host='0.0.0.0')
