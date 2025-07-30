import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_rf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

gejala_list = [
    "nyeri_saat_mengunyah", "gusi_bengkak", "bau_mulut", "pendarahan_gusi",
    "gigi_ngilu", "nanah_di_gusi", "warna_gusi_kemerahan", "gigi_berlubang",
    "pembengkakan_pipi", "demam", "rasa_tidak_enak_di_mulut", "lidah_pahit",
    "gusi_mundur", "nafsu_makan_menurun", "sakit_saat_minum_dingin",
    "rahang_nyeri", "sakit_gigi_berdenyut", "kesulitan_mengunyah",
    "gigi_longgar", "tidak_nyaman_di_rahang"
]

st.title("🦷 Prediksi Penyakit Gigi Berdasarkan Gejala")

input_gejala = [st.checkbox(g.replace("_", " ").capitalize()) for g in gejala_list]

if st.button("Prediksi"):
    df_input = pd.DataFrame([input_gejala], columns=gejala_list)
    pred = model.predict(df_input)[0]
    penyakit = label_encoder.inverse_transform([pred])[0]
    st.success(f"Hasil Prediksi: **{penyakit}**")
