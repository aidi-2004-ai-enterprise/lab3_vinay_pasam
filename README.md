# lab3_vinay_pasam
# Penguin Species Classification API

🎥 **Demo Video:** [`recording/demo.mp4`](recording/demo.mp4)

---

## 📌 Overview

This project builds a machine learning API using **FastAPI** to classify penguin species based on physical and categorical features. The classification model is trained using **XGBoost**, and label encoding is used for preprocessing.

---

## 📁 Project Structure

```
lab3_vinay_pasam/
├── app/
│   ├── main.py              # FastAPI app
│   └── data/
│       ├── xgb_penguin_model.json
│       └── label_encoder.pkl
├── train.py                 # Model training script
├── recording/
│   └── demo.mp4             # Demo video (local)
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
├── pyproject.toml
├── .gitignore
├── .python-version

```

---

## 🚀 How to Run the API

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open in browser:**
   ```
   http://127.0.0.1:8000/docs
   ```

---

## 🧠 Model Info

- **Model file:** `xgb_penguin_model.json`
- **Preprocessing:** One-hot encoding for `sex` and `island`, dropped `year`
- **Target:** Penguin species (`Adelie`, `Chinstrap`, `Gentoo`)

---

## 📨 Sample Valid Request (200 OK)

```json
{
  "bill_length_mm": 43.2,
  "bill_depth_mm": 17.5,
  "flipper_length_mm": 180,
  "body_mass_g": 3700,
  "sex": "Male",
  "island": "Biscoe"
}
```

**✅ Response:**
```json
{
  "predicted_species": "Adelie"
}
```

---

## ❌ Sample Invalid Request (422 Unprocessable Entity)

```json
{
  "bill_length_mm": 45.0,
  "bill_depth_mm": 15.0,
  "flipper_length_mm": 200,
  "body_mass_g": 5000,
  "sex": "Unknown",
  "island": "India"
}
```

**🚫 Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "sex"],
      "msg": "value is not a valid enumeration member; permitted: 'Male', 'Female'",
      "type": "type_error.enum"
    },
    {
      "loc": ["body", "island"],
      "msg": "value is not a valid enumeration member; permitted: 'Biscoe', 'Dream', 'Torgersen'",
      "type": "type_error.enum"
    }
  ]
}
```

---

## 📦 Dependencies

- `fastapi`
- `uvicorn`
- `xgboost`
- `scikit-learn`
- `pydantic`
- `pandas`

---

## 👤 Author

**Vinay Pasam**  
📧 vinaychoudary741@gmail.com

---

> 🧪 Built for Lab 3 - Penguins Classification using XGBoost & FastAPI  
> 📂 Includes video demo: `recording/demo.mp4`
