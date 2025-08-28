## Install streamlit if you dont have by the following command. 
%pip install streamlit

import streamlit as st
import numpy as np
import joblib
# Load model
model = joblib.load("phishing_model.pkl")


