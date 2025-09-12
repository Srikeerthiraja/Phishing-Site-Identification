## Install streamlit if you dont have by the following command. 
%pip install streamlit

import streamlit as st
import joblib
import numpy as np
import re
from urllib.parse import urlparse

# Load model
model = joblib.load("best_lexical_pipeline.pkl")
feature_order = joblib.load("lexical_features_list.pkl")

def extract_features(url):
    parsed = urlparse(url if url.startswith(("http://", "https://")) else "http://" + url)
    u = url.lower()

    f = {
        # 1. Having IP Address
        "having_IP_Address": 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', parsed.netloc) else 0,
        # 2. URL Length
        "URL_Length": len(u),
        # 3. Shortening service
        "Shortining_Service": 1 if re.search(r"(bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co|tinyurl\.com)", u) else 0,
        # 4. Having @ symbol
        "having_At_Symbol": 1 if "@" in u else 0,
        # 5. Double slash redirecting
        "double_slash_redirecting": 1 if u.count("//") > 1 else 0,
        # 6. Prefix-Suffix
        "Prefix_Suffix": 1 if "-" in parsed.netloc else 0,
        # 7. Having Sub Domain
        "having_Sub_Domain": 1 if len(parsed.netloc.split(".")) > 2 else 0,
        # 8. HTTPS token
        "HTTPS_token": 1 if "https" in parsed.netloc else 0,
        # 9. URL of Anchor (heuristic)
        "URL_of_Anchor": 1 if "#" in u or "javascript" in u else 0,
        # 10. Links in tags (approx)
        "Links_in_tags": 1 if "<a" in u or "<link" in u else 0,
        # 11. SFH (Suspicious form handler)
        "SFH": 1 if "action=" in u or "form" in u else 0,
        # 12. Submitting to email
        "Submitting_to_email": 1 if "mailto:" in u else 0,
        # 13. Abnormal URL (long domain)
        "Abnormal_URL": 1 if len(parsed.netloc) > 30 else 0,
        # 14. Redirect
        "Redirect": 1 if "redirect" in u or "redir" in u else 0,
        # 15. onmouseover
        "on_mouseover": 1 if "onmouseover" in u else 0,
        # 16. RightClick
        "RightClick": 1 if "rightclick" in u else 0,
        # 17. PopUp Window
        "popUpWidnow": 1 if "popup" in u else 0,
        # 18. Iframe
        "Iframe": 1 if "iframe" in u else 0,
    }
    return np.array([f.get(col, 0) for col in feature_order]).reshape(1, -1)

st.title("🔒 Phishing URL Detection App")
st.write("Enter a URL to check if it's **Legitimate or Phishing**")
url_input = st.text_input("Enter URL here:")

if st.button("Check URL"):
    if url_input:
        features = extract_features(url_input)
        try:
            prediction = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]

            if prediction == 1:
                st.error(f" This looks like a **Phishing URL** (Confidence: {prob:.2f})")
            else:
                st.success(f"This looks **Legitimate** (Confidence: {1-prob:.2f})")
        except Exception as e:
            st.error(f"Model failed: {e}")
    else:
        st.warning("Please enter a URL first!")



## run the streamlit using the command
- streamlit run app.py


