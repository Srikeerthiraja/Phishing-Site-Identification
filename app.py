## Install streamlit if you dont have by the following command. 
%pip install streamlit

import streamlit as st
import numpy as np
import joblib
from tld import get_tld

def having_ip_address(url): return 1 if any(ch.isdigit() for ch in url) else 0
def abnormal_url(url): return 0
def count_dot(url): return url.count('.')
def count_www(url): return url.count('www')
def count_atrate(url): return url.count('@')
def no_of_dir(url): return url.count('/')
def no_of_embed(url): return url.count('//')
def shortening_service(url): return 1 if "bit.ly" in url or "tinyurl" in url else 0
def count_https(url): return url.count('https')
def count_http(url): return url.count('http')
def count_per(url): return url.count('%')
def count_ques(url): return url.count('?')
def count_hyphen(url): return url.count('-')
def count_equal(url): return url.count('=')
def url_length(url): return len(url)
def hostname_length(url): return len(url.split('/')[2]) if '//' in url else len(url)
def suspicious_words(url):
    words = ['secure','account','update','login','verify']
    return any(w in url.lower() for w in words)
def digit_count(url): return sum(ch.isdigit() for ch in url)
def letter_count(url): return sum(ch.isalpha() for ch in url)
def fd_length(url): return len(url.split('/')[0])
def tld_length(tld): return len(tld) if tld else 0

# Feature Extraction
def main(url):
    status = [
        having_ip_address(url),
        abnormal_url(url),
        count_dot(url),
        count_www(url),
        count_atrate(url),
        no_of_dir(url),
        no_of_embed(url),
        shortening_service(url),
        count_https(url),
        count_http(url),
        count_per(url),
        count_ques(url),
        count_hyphen(url),
        count_equal(url),
        url_length(url),
        hostname_length(url),
        suspicious_words(url),
        digit_count(url),
        letter_count(url),
        fd_length(url)
    ]
    tld = get_tld(url, fail_silently=True)
    status.append(tld_length(tld))
    return status

# Prediction
def get_prediction_from_url(test_url, model):
    features_test = np.array(main(test_url)).reshape((1, -1))
    pred = model.predict(features_test)
    labels = {
        0: "SAFE ✅",
        1: "DEFACEMENT ",
        2: "PHISHING ",
        3: "MALWARE "
    }
    return labels.get(int(pred[0]), "UNKNOWN")

# Streamlit App UI

def run_app():
    st.set_page_config(page_title="URL Threat Detector", page_icon="🌐", layout="centered")

    st.markdown("""
        <style>
            .main {
                background-color: #0E1117;
                color: white;
                font-family: 'Poppins', sans-serif;
            }
            h1 {
                text-align: center;
                color: #4FC3F7;
                font-weight: 700;
            }
            .stButton>button {
                background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%);
                color: white;
                border-radius: 10px;
                padding: 0.6em 1.5em;
                font-size: 1.1em;
                font-weight: 600;
                border: none;
            }
            .stTextInput>div>div>input {
                background-color: 1E1E1E;
                color: white;
                border-radius: 8px;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1> URL Threat Detection App</h1>", unsafe_allow_html=True)
    st.write("Analyze a URL to check whether it’s **SAFE**, **DEFACEMENT**, **PHISHING**, or **MALWARE**.")

    url = st.text_input("🌐 Enter a URL to scan:", placeholder="e.g., http://example.com/login")

    try:
        model = joblib.load("lgb_model.pkl")
    except:
        st.warning("Model not found! Please ensure **lgb_model.pkl** is in the same folder.")
        return
    if st.button("Check URL"):
        if url.strip() == "":
            st.info("Please enter a valid URL before predicting.")
        else:
            with st.spinner("🔎 Scanning the URL..."):
                result = get_prediction_from_url(url, model)

            if "SAFE" in result:
                st.success(f" The URL is classified as: **{result}**")
            elif "DEFACEMENT" in result:
                st.warning(f"The URL is classified as: **{result}**")
            elif "PHISHING" in result:
                st.error(f"The URL is classified as: **{result}**")
            elif "MALWARE" in result:
                st.error(f"The URL is classified as: **{result}**")

            st.markdown("---")
            st.info(" Tip: Always double-check URLs that request your login details or financial information!")


if __name__ == "__main__":
    run_app()



## run the streamlit using the command
- streamlit run app.py


