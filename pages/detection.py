import streamlit as st
import os
import spacy
from spacy import displacy
from spacy.tokens import Span
import tempfile
from bs4 import BeautifulSoup
import stanza
import re
import joblib

# Menambahkan styling CSS untuk latar belakang dan tombol
st.markdown("""
    <style>
        .stButton > button {
            background-color: #007bff;  /* Biru */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #0056b3;  /* Biru lebih gelap saat hover */
            border-color: white;
            color: white;
        }
        .stTextInput>div>div>input {
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            height: 200px;
            color: black;
            border-radius: 3px;
            border: 2px solid #90b7e0;
            text-align: center;  /* Mengatur teks agar berada di tengah horizontal */
            vertical-align: middle;  /* Mengatur teks agar berada di tengah vertikal */
            display: flex;
            justify-content: center;
            align-items: center;
        }
        #balinese-person-entity>div>span {
            font-size: 60px;    
        }
    </style>
""", unsafe_allow_html=True)

def remove_punctuation(text):
    """
    Fungsi ini digunakan untuk membersihkan kata dari tanda baca menggunakan pustaka regex
    [^\w\s] artinya mencocokkan seluruh character yang bukan huruf (\w) atau spasi (\s)
    """
    return re.sub(r'[^\w\s]', '', text)



def word2features(sentence, i):
    word = sentence[i][0]
    pos = sentence[i][1]    
    features = {
        'word': word,
        'pos': pos,
        'is_first': i == 0, #if the word is a first word
        'is_last': i == len(sentence) - 1,  #if the word is a last word
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,      #word is in uppercase
        'is_all_lower': word.lower() == word,      #word is in lowercase
         #prefix of the word
        'prefix-1': word[0],   
        'prefix-2': word[:2],
        'prefix-3': word[:3],
         #suffix of the word
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
         #extracting previous word
        'prev_word': '' if i == 0 else sentence[i-1][0],
        'prev_pos': '' if i == 0 else sentence[i-1][1],
         #extracting next word
        'next_word': '' if i == len(sentence)-1 else sentence[i+1][0],
        'next_pos': '' if i == len(sentence)-1 else sentence[i+1][1],
        'is_numeric': word.isdigit(),  #if word is in numeric
        'capitals_inside': word[1:].lower() != word[1:]
    }
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def predict(sentence, model_ner, nlp):
    """
    fungsi ini digunakan untuk mencari nilai ner dari input user
    sentence => string kalimat dari user
    model_ner => model pycrfsuite yang telah dilatih
    model_pos => model pycrfsuite yang telah dilatih
    emb_model => model embedding yang telah dilatih
    """

    sentence_split = sentence.split()
    
    doc = nlp(sentence)
    pos = [word.pos for sent in doc.sentences for word in sent.words]

    ner = model_ner.predict([sent2features(list(zip(sentence_split, pos)))])

    return ner


def cust_entities(sent, tagger, nlp):
    """
    fungsi ini digunakan untuk memanggil fungsi predict dan membentuk custom entities 
    pada display spacy dengan menyesuaikan data diperlukan untuk visualisasi menggunakan 
    displacy. 
    index_from => index awal suatu entitas dimulai, nilai bisa lebih dari satu (B-)
    index_end => index akhir suatu entitas berakhir, nilai bisa lebih dari satu (B- or I-)
    """

    entity_tags = predict(sent, tagger, nlp)
    print(entity_tags[0])
    nlp = spacy.blank("id") 
    doc = nlp(sent)
    tokens = sent.split()

    custom_ents = []
    start = None    
    end = None
    ent_label = None

    for i, tag in enumerate(entity_tags[0]):
        if tag.startswith("B-"):
            # simpan entitas sebelumnya jika ada
            if start is not None:
                word_entity = ' '.join(tokens[start:end])
                custom_ents.append((start, end, ent_label, word_entity))

            # mulai entitas baru
            start = i
            end = i + 1
            ent_label = tag[2:]  # ambil label tanpa "B-"
        
        elif tag.startswith("I-") and ent_label == tag[2:]:
            # lanjutan dari entitas yang sedang berlangsung
            end += 1
        else:
            # tag O atau I-label yang tidak sesuai
            if start is not None:
                word_entity = ' '.join(tokens[start:end])
                custom_ents.append((start, end, ent_label, word_entity))
                start = None
                end = None
                ent_label = None

    # cek entitas terakhir
    if start is not None:
        word_entity = ' '.join(tokens[start:end])
        custom_ents.append((start, end, ent_label, word_entity))

    print(custom_ents)
    return custom_ents, doc

def visualize(custom_ents, doc):
    """
    fungsi ini digunakan untuk visualisasi ke web page menggunakan displacy
    """
    ents = [Span(doc, start, end, label=label) for start, end, label, _ in custom_ents]
    doc.ents = ents

    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmpfile:
        html = displacy.render(doc, style="ent", page=True)
        tmpfile.write(html)
        tmpfile.close()

        with open(tmpfile.name, "r", encoding="utf-8") as f:
            html_content = f.read()

            soup = BeautifulSoup(html_content, 'html.parser')
            entities_div = soup.find_all("div", class_="entities")

            cleaned_html = ''.join(str(div) for div in entities_div)

            st.markdown(cleaned_html, unsafe_allow_html=True)

# Streamlit UI

st.title("SKILL ENTITY DETECTION")

nlp = stanza.Pipeline('id', processors='tokenize,pos', use_gpu=False, tokenize_pretokenized=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODEL_PATH = os.path.join(PARENT_DIR,'assets', 'best_crf_model.pkl')
crf = joblib.load(MODEL_PATH)

# Input text from the user
user_input = st.text_area(
label="Masukkan Teks Anda",               # label terlihat
placeholder="Contoh: Saya suka belajar pemrograman",  # teks abu-abu sebagai petunjuk
height=150                                 # opsional: tinggi textarea
)
clean_input = remove_punctuation(user_input)

# Create a container for the button on the right
col1, col2 = st.columns([8, 1])  # Two columns, with second column smaller
with col2:
    button = st.button("Cek")
if button:
    if user_input:
        # Predict entities from the input text
        with st.spinner("Memproses... Harap tunggu"):
            custom_ents, doc = cust_entities(clean_input, crf, nlp)

        # Display results
        if custom_ents:
            st.write(" ")
            st.markdown("---")
            visualize(custom_ents, doc)
        else:
            st.write("Tidak ada entitas yang terdeteksi.")
    else:
        st.write("Masukkan teks terlebih dahulu.")