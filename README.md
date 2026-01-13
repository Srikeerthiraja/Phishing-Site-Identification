# Phishing-Site-Identification

## Problem Statement
Phishing websites are a major cybersecurity threat, tricking users into revealing sensitive
information such as passwords and financial data. Traditional blacklist-based approaches fail
to detect newly generated phishing URLs, making real-time detection challenging.

The goal of this project is to build a machine learning system that can automatically classify
URLs as legitimate or phishing based on their lexical characteristics, with an emphasis on
high recall for phishing sites to minimize security risks.

## Solution Overview
This project implements a machine learning-based phishing URL detection system using
lexical feature engineering and ensemble learning models.

The system extracts statistical and textual features directly from URLs, trains multiple
classification models, and selects the best-performing model (XGBoost) based on recall
and overall generalization. The final model is deployed as a real-time Streamlit web
application for interactive testing.


## Features
- Extracts multiple lexical features from URLs (length, digits ratio, suspicious keywords, etc.)
- Trains ML models (RandomForest, XGBoost, Light GBM Classifier)
- Achieves **~85% accuracy** and **0.99 Recall for Legit and 0.82 for Phishing sites**
- Interactive **Streamlit app** for real-time URL testing
- Packaged ML pipeline (`.pkl`) for reusability

## Modeling Approach
- Performed lexical feature extraction from URLs (length, digit ratio, special characters,
  suspicious keywords, entropy).
- Trained and compared Random Forest, LightGBM, and XGBoost classifiers.
- Evaluated models using accuracy, precision, recall, and F1-score with special focus on
  phishing recall due to the high cost of false negatives.
- Selected XGBoost as the final model due to its superior recall and robustness to feature
  interactions.

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

High phishing recall ensures better protection against malicious URLs.

## Design Decisions & Trade-offs

### Lexical Features vs Website Content Analysis
Lexical features were chosen because they enable fast classification without fetching website
content. This improves inference speed and avoids dependency on external network calls.
The trade-off is reduced context compared to HTML or JavaScript-based analysis.

### XGBoost vs Deep Learning Models
XGBoost was selected for its strong performance on tabular data, faster training, and easier
interpretability. Deep learning models could potentially achieve higher accuracy but would
require larger datasets and more computational resources.

### High Recall vs Precision
The system prioritizes high recall for phishing URLs to reduce false negatives, even at the
cost of occasionally misclassifying legitimate URLs as phishing.

### Streamlit Deployment vs Production API
Streamlit enables rapid prototyping and easy interaction. However, it is less suitable for
high-throughput production environments compared to a REST API-based service.

## Future Improvements
- Explore deep learning approaches (LSTM, CNN) on raw URL sequences
- Use word embeddings for domain and path tokenization
- Integrate the model into a browser extension for real-time phishing alerts

