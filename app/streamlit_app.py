"""
streamlit_app.py
-----------------
Interface graphique : l'utilisateur dépose une image, l'app affiche
l'image, appelle predict.py, et montre la classe prédite + la confiance.

Utilisation :
    streamlit run app/streamlit_app.py
"""

import os
import sys
import tempfile

import streamlit as st

# app/ et src/ sont deux dossiers frères -> on ajoute src/ au chemin Python
# pour pouvoir importer predict.py et config.py sans dupliquer le code.
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)

from predict import predict_image  # noqa: E402  (import après modification de sys.path, volontaire)
from config import MODEL_PATH  # noqa: E402


st.set_page_config(page_title="Classification de déchets", page_icon="♻️")

st.title("♻️ Classification de déchets")
st.write("Dépose une image de déchet, le modèle prédit s'il est **organique** ou **recyclable**.")

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Modèle introuvable à {MODEL_PATH}. "
        "Lance d'abord `python src/train_model.py` pour l'entraîner."
    )
    st.stop()

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Image envoyée", use_container_width=True)

    # predict_image attend un chemin de fichier -> on écrit l'upload dans un
    # fichier temporaire le temps de la prédiction.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Analyse en cours..."):
            label, confidence = predict_image(tmp_path)
    finally:
        os.remove(tmp_path)

    with col2:
        if label == "organic":
            st.success(f"### 🌱 Organique")
        else:
            st.info(f"### ♻️ Recyclable")
        st.metric("Confiance", f"{confidence * 100:.1f}%")
        st.progress(confidence)
