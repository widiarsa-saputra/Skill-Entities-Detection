import streamlit as st
import os
import pandas as pd
from PIL import Image

st.title('Dataset dan Performa Model')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
DATA_PATH = os.path.join(PARENT_DIR, 'assets', 'data_train_skill_ner.pkl')

try:
    df = pd.read_pickle(DATA_PATH)
except FileNotFoundError:
    df = None
    st.error(f"File tidak ditemukan di: {DATA_PATH}")

if df is not None:
    st.write("Dataset:")
    st.dataframe(df, use_container_width=True)
    st.markdown(f"Jumlah baris: `{len(df)}`")
else:
    st.warning("Dataset belum dimuat.") 


st.title("Distribusi Kelas")
IMG_DATA_PATH = os.path.join(PARENT_DIR,'assets','all_labels.png')
IMG_DATA_PATH2 = os.path.join(PARENT_DIR,'assets','entities.png')

image = Image.open(IMG_DATA_PATH)
image2 = Image.open(IMG_DATA_PATH2)

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Seluruh Entitas", use_container_width=True)
with col2:
    st.image(image2, caption="Seluruh Entitas Bernama", use_container_width=True)


st.title('Hasil Grid-Search')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PARENT_DIR, 'assets','grid_search_crf_report.csv')

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    df = None
    st.error(f"File tidak ditemukan di: {DATA_PATH}")

if df is not None:
    st.write("Hasil :")
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Dataset belum dimuat.") 

st.title('Best Model Report')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PARENT_DIR, 'assets','hasil.csv')

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    df = None
    st.error(f"File tidak ditemukan di: {DATA_PATH}")

if df is not None:
    st.write("Evaluation:")
    st.data_editor(df, use_container_width=True, hide_index=True)

else:
    st.warning("Dataset belum dimuat.") 

st.title("Best Model Confusion Matrix")
IMG_DATA_PATH = os.path.join(PARENT_DIR,'assets','confusion-matriks.png')
print("Image path:", IMG_DATA_PATH)
image = Image.open(IMG_DATA_PATH)
st.image(image, caption=" ", use_container_width=True)