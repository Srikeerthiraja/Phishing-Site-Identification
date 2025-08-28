# Phishing-Site-Identification
## Overview
This project detects whether a given URL is **legitimate or phishing** using **machine learning**.  
It combines **lexical feature engineering** with a **Random Forest** and is deployed as an interactive web app using **Streamlit**.
## Features
- Extracts multiple lexical features from URLs (length, digits ratio, suspicious keywords, etc.)
- Trains ML models (RandomForest, XGBoost)
- Achieves **~97% accuracy** and **0.99 ROC AUC**
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
| Accuracy      | 90%+  |
| F1 Score      | 0.97  |
| ROC AUC       | 0.99  |
## Future Work
- Use Deep Learning
- Try LSTM / CNN models on raw URLs instead of handcrafted features.
- Use word embeddings for domain/URL tokens.
- Integrate with a browser extension to flag phishing websites instantly.
