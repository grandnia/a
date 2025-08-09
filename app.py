import streamlit as st
import pandas as pd
import cloudpickle
import joblib
import os

st.title("Prediksi Profit Menu Restoran")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, "pipeline_rfnew.pkl")
encoder_path = os.path.join(BASE_DIR, "target_encoder.pkl")

# Load pipeline
with open(pipeline_path, 'rb') as f:
    pipeline = cloudpickle.load(f)

# Load label encoder (jika diperlukan)
label_encoder = joblib.load(encoder_path)

# Input user
menu_item = st.text_input('Nama Menu', 'Nasi Goreng')
restaurant_id = st.text_input('ID Restoran', 'R001')
price = st.number_input('Harga Jual per Produk (Rp)', min_value=0, value=25000)
menu_category = st.selectbox('Kategori Menu', ['Makanan', 'Minuman', 'Dessert'])
ingredients = st.text_area('Bahan-bahan', 'Nasi, Telur, Ayam, Kecap')

input_data = pd.DataFrame([{
    'MenuItem': menu_item,
    'RestaurantID': restaurant_id,
    'Price': price,
    'MenuCategory': menu_category,
    'Ingredients': ingredients
}])

if st.button('Prediksi Profit'):
    try:
        prediksi = pipeline.predict(input_data)
        st.success(f"Estimasi profit: Rp {prediksi[0]:,.2f}")
    except Exception as e:
        st.error(f"Error saat prediksi: {e}")
