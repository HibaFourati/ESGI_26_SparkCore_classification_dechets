"""
predict.py
----------
Fonction de prédiction réutilisable : charge le modèle entraîné une fois,
prétraite une image, retourne la classe prédite + la confiance.

Conçu pour être importé par app/streamlit_app.py, mais utilisable aussi en
ligne de commande :
    python predict.py --image /chemin/vers/image.jpg
"""

import argparse

import numpy as np
import tensorflow as tf

from config import MODEL_PATH, IMG_SIZE, LABELS

_model_cache = None  # évite de recharger le modèle à chaque appel (utile pour Streamlit)


def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = tf.keras.models.load_model(MODEL_PATH)
    return _model_cache


def predict_image(image_path: str):
    """
    Prend un chemin d'image, retourne (label_prédit: str, confiance: float entre 0 et 1).
    """
    model = load_model()

    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0  # même normalisation qu'à l'entraînement
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3) : le modèle attend un batch

    score = float(model.predict(img_array, verbose=0)[0][0])

    # Rappel : classes triées alphabétiquement par Keras -> organic=0, recyclable=1
    predicted_label = "recyclable" if score >= 0.5 else "organic"
    confidence = score if score >= 0.5 else 1 - score

    return predicted_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    label, confidence = predict_image(args.image)
    print(f"[PREDICTION] Classe : {label} | Confiance : {confidence * 100:.2f}%")
