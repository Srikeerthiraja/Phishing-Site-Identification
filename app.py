## Install streamlit if you dont have by the following command. 
%pip install streamlit

import streamlit as st
import joblib
import numpy as np
import re, urllib.parse
model = joblib.load("phishing_model.pkl")

def extract_features(url):
    url_length = len(url)
    dots = url.count(".")
    has_at = 1 if "@" in url else 0
    hyphens = url.count("-")
    https = 1 if url.startswith("https") else 0
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    has_ip = 1 if re.search(ip_pattern, url) else 0
    query = urllib.parse.urlparse(url).query
    num_params = len(query.split("&")) if query else 0

    return np.array([url_length, dots, has_at, hyphens, https, has_ip, num_params]).reshape(1, -1)

st.title("🔒 Phishing URL Detection App")
st.write("Enter a URL to check if it's **Legitimate or Phishing**")

url_input = st.text_input("Enter URL here:")

if st.button("Check URL"):
    if url_input:
        features = extract_features(url_input)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if prediction == 1:
            st.error(f"⚠️ This looks like a **Phishing URL** (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ This looks **Legitimate** (Confidence: {1-prob:.2f})")
    else:
        st.warning("Please enter a URL first!")


## run the streamlit using the command
- streamlit run app.py


