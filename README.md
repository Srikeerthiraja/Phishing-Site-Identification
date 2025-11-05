# Phishing-Site-Identification
## Overview
This project detects whether a given URL is **legitimate or phishing** using **machine learning**.  
It combines **lexical feature engineering** with a **XGBoost** and is deployed as an interactive web app using **Streamlit**.
## Features
- Extracts multiple lexical features from URLs (length, digits ratio, suspicious keywords, etc.)
- Trains ML models (RandomForest, XGBoost, Light GBM Classifier)
- Achieves **~85% accuracy** and **0.99 Recall for Legit and 0.82 for Phishing sites**
- Interactive **Streamlit app** for real-time URL testing
- Packaged ML pipeline (`.pkl`) for reusability
## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn (EDA + visualizations)
- Streamlit (deployment)
## 📊 Results
| Metric        | Score |
|---------------|-------|
| Accuracy      | 85    |
| F1 Score      | 0.97  |
| Recall        | 0.99  |
## Future Work
- Use Deep Learning
- Try LSTM / CNN models on raw URLs instead of handcrafted features.
- Use word embeddings for domain/URL tokens.
- Integrate with a browser extension to flag phishing websites instantly.
