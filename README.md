# InsureIQ — Insurance Cost Predictor

A full-stack web application for predicting insurance charges using a Gradient Boosting machine learning model.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)
```bash
cd backend
python train_model.py
```

### 3. Run the server
```bash
cd backend
python app.py
```

### 4. Open in browser
Navigate to: **http://localhost:5000**

---

## 📁 Project Structure

```
insurance_app/
├── backend/
│   ├── app.py              ← Flask backend (main server)
│   ├── train_model.py      ← Train & save the ML model
│   ├── insurance.csv       ← Original dataset (1338 records)
│   ├── insurance_model.pkl ← Trained ML pipeline (auto-generated)
│   ├── insurance_app.db    ← SQLite database (auto-created)
│   └── new_inputs.csv      ← Collected prediction inputs (auto-created)
├── frontend/
│   └── templates/
│       ├── login.html      ← Login/Register page
│       └── dashboard.html  ← Main dashboard
└── requirements.txt
```

---

## 🔑 Features

### Authentication
- **Register** with email + password
- **Login** with session tokens (8-hour expiry)
- **Password strength meter** (Weak / Fair / Good / Strong)
- **CAPTCHA** anti-robot check on all auth forms
- Passwords stored as **PBKDF2-SHA256** hashes with unique salts

### Prediction
- Input: Age, Gender, BMI, Children, Smoker status, Region
- Model: **Gradient Boosting Regressor** (sklearn) — equivalent to XGBoost
  - 300 estimators, learning rate 0.05, max depth 5
  - Train R²: ~0.98 | Test R²: ~0.83
- Every prediction is saved to the **SQLite database** and **new_inputs.csv**

### Data Management
- **Download new_insurance_inputs.csv** — all new prediction inputs
- **Download updated_insurance_dataset.csv** — original + new inputs merged
- **Retrain model** on-demand from the dashboard

### Dashboard
- Stats: total predictions, average charge, highest estimate, dataset records
- Prediction history table (last 20 records)
- Download center for CSV exports

---

## 🗄️ Database Schema

**users** — email, hashed password, salt, created_at  
**sessions** — user_id, token, expires_at  
**predictions** — user_id, all input fields, predicted_charge, created_at

---

## 🔧 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/register` | ❌ | Create account |
| POST | `/api/login` | ❌ | Login, get token |
| POST | `/api/logout` | ✅ | Invalidate session |
| POST | `/api/predict` | ✅ | Get insurance estimate |
| GET | `/api/history` | ✅ | Last 20 predictions |
| GET | `/api/stats` | ✅ | Dashboard stats |
| GET | `/api/download/new-inputs` | ✅ | Download new_inputs.csv |
| GET | `/api/download/updated-dataset` | ✅ | Download merged dataset |
| POST | `/api/retrain` | ✅ | Retrain ML model |

---

## 📊 Dataset

The model is trained on `insurance.csv` with 1338 records and these features:
- **age** — integer
- **sex** — male / female
- **bmi** — float (body mass index)
- **children** — integer (dependents)
- **smoker** — yes / no
- **region** — southwest / southeast / northwest / northeast
- **charges** — float (target variable, annual insurance charge in USD)
