import streamlit as st

st.set_page_config(
    page_title="homepage    "
)

def main():
    st.title("📌 Aplikasi Deteksi Skill dengan NERSkill.id")

    st.markdown("""
    ### Deskripsi Aplikasi Deteksi Skill
    Aplikasi ini dirancang untuk melakukan **Named Entity Recognition (NER)** guna mendeteksi entitas berupa **skills atau keahlian** dalam teks berbahasa Indonesia. Aplikasi menggunakan model **Conditional Random Fields (CRF)** yang telah dilatih menggunakan dataset **NERSkill.id**.

    Pengguna cukup memasukkan teks (misalnya: deskripsi pekerjaan, CV, atau postingan lowongan kerja), dan sistem akan menandai kata-kata yang termasuk dalam kategori skill atau entitas terkait lainnya.
    """)

    st.markdown("""
    ### 📊 Tentang Dataset: NERSkill.id
    **NERSkill.id** adalah dataset anotasi NER berbahasa Indonesia yang berfokus pada domain ketenagakerjaan, khususnya untuk mendeteksi:

    - **Skills** atau kemampuan teknis/non-teknis
    - **Tools** atau alat/teknologi yang digunakan
    - **Job Title** atau jabatan pekerjaan
    - **Certifications** atau sertifikasi keahlian
    - Dan beberapa entitas terkait lainnya
    """)

    st.markdown("""
    ### 🏷️ Pembagian Label dalam NERSkill.id
    Label dalam dataset ini mengikuti format BIO (Begin, Inside, Outside), yang berarti:

    - **B-XXX**: Awal dari entitas bertipe XXX  
    - **I-XXX**: Bagian lanjutan dari entitas bertipe XXX  
    - **O**: Bukan bagian dari entitas mana pun
    """)

    st.markdown("#### Contoh Label Umum:")
    st.table({
        "Label": ["B-SKILL", "I-SKILL", "B-JOB", "I-JOB", "B-TOOL", "I-TOOL", "B-CERT", "I-CERT", "O"],
        "Deskripsi": [
            "Awal dari nama skill", "Lanjutan dari skill multi-kata",
            "Awal dari jabatan kerja", "Lanjutan dari jabatan kerja",
            "Awal dari nama alat/teknologi", "Lanjutan dari nama alat",
            "Awal dari nama sertifikasi", "Lanjutan dari sertifikasi",
            "Bukan bagian dari entitas apa pun"
        ]
    })

    st.markdown("""
    ### 🔍 Contoh Kalimat dan Label

    **Kalimat**:  
    “Dia menguasai Python dan Machine Learning serta bekerja sebagai Data Scientist.”

    **Label Token**:
    ```
    Dia          O  
    menguasai    O  
    Python       B-SKILL  
    dan          O  
    Machine      B-SKILL  
    Learning     I-SKILL  
    serta        O  
    bekerja      O  
    sebagai      O  
    Data         B-JOB  
    Scientist    I-JOB  
    ```
    """)

    st.markdown("""
    Aplikasi ini cocok untuk digunakan dalam analisis data tenaga kerja, penyaringan CV, dan sistem rekomendasi pekerjaan.
    """)

if __name__ == "__main__":
    main()
