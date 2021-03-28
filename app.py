#app.py
# 
import app1
import app2
import streamlit as st
from PIL import Image
image = Image.open('cars.jpg')
st.sidebar.image(image,use_column_width=True)
PAGES = {
    "Predict Customer Loyalty": app1,
    "Data Analytics": app2
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()