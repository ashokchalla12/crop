# 🌾 OptiCrop — Smart Agricultural Production Optimization Engine

An intelligent agricultural decision support system that recommends the best crop based on soil nutrients (N, P, K) and environmental conditions (temperature, humidity, pH, rainfall).

![Python](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-orange)

## ✨ Features

- **Smart Crop Recommendation** — AI-powered predictions with confidence scores
- **Crop Suitability Assessment** — Compare field conditions vs ideal requirements
- **Research Dashboard** — Interactive Plotly analytics and visualizations
- **Dataset Explorer** — Search, filter, sort, and download data
- **ML Model Training** — Train & compare 6 algorithms, auto-save best model
- **PDF Export** — Download prediction reports
- **Dark Mode** — Toggle between light and dark themes
- **Prediction History** — Track last 20 predictions

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Frontend | Streamlit |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Model Storage | Joblib, Pickle |

## 📁 Project Structure

```
OptiCrop/
├── app.py                  # Main Streamlit entry point
├── train_model.py          # ML training script
├── requirements.txt
├── README.md
├── model/
│   ├── crop_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── metrics.json
├── data/
│   └── Crop_recommendation.csv
├── pages/
│   ├── Home.py
│   ├── Prediction.py
│   ├── Suitability.py
│   ├── Dashboard.py
│   ├── Dataset.py
│   └── About.py
├── utils/
│   ├── helper.py
│   ├── charts.py
│   └── preprocessing.py
└── assets/
    ├── logo.png
    ├── background.jpg
    └── crop_images/
```

## 🚀 Quick Start

### 1. Clone / Navigate to project

```bash
cd OptiCrop
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML model

```bash
python train_model.py
```

### 5. Run the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| N | Nitrogen content in soil |
| P | Phosphorous content in soil |
| K | Potassium content in soil |
| temperature | Temperature (°C) |
| humidity | Relative humidity (%) |
| ph | Soil pH |
| rainfall | Rainfall (mm) |
| label | Crop name (22 classes) |

**Supported Crops:** Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 🤖 Machine Learning

Six algorithms are trained and compared:

- Random Forest
- Decision Tree
- SVM
- KNN
- Naive Bayes
- Logistic Regression

Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix

The best-performing model is automatically saved to `model/crop_model.pkl`.

## 📸 Screenshots

Place application screenshots in the `screenshots/` folder.

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

OptiCrop Development Team — Smart Agricultural Production Optimization Engine v1.0
