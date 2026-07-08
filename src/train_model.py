"""
train_model.py
---------------
Entraînement d'un CNN simple (Python/TensorFlow-Keras, PAS Spark) sur les
images préparées par spark_pipeline.py (output/stage3_ready/).

Architecture volontairement simple, pour bien comprendre chaque brique :
    [Conv2D -> MaxPooling2D] x3   <- extraction de features visuelles
    Flatten                       <- mise à plat en un vecteur
    Dense(128) -> Dropout         <- "réflexion", couche de décision
    Dense(1, sigmoid)             <- sortie : probabilité "recyclable"

Utilisation :
    python train_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import (
    STAGE3_READY_TRAIN, STAGE3_READY_TEST,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_DIR, MODEL_PATH,
)


def load_datasets():
    """
    image_dataset_from_directory lit directement l'arborescence de dossiers
    (organic/, recyclable/) produite par le stage 3 du pipeline Spark, et
    déduit automatiquement les labels depuis les noms de sous-dossiers.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        STAGE3_READY_TRAIN,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",   # 2 classes -> une seule valeur 0 ou 1 par image
        validation_split=0.15,  # 15% du train mis de côté pour la validation
        subset="training",
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        STAGE3_READY_TRAIN,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        validation_split=0.15,
        subset="validation",
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        STAGE3_READY_TEST,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,  # pas besoin de mélanger pour l'évaluation finale
    )

    print(f"[INFO] Classes détectées (ordre alphabétique) : {train_ds.class_names}")
    # -> ['organic', 'recyclable'] : organic=0, recyclable=1 pour la sortie sigmoid

    # Normalisation des pixels : [0,255] -> [0,1]. Les réseaux de neurones
    # convergent beaucoup mieux avec des valeurs d'entrée petites et centrées.
    normalization = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalization(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_simple_cnn():
    """
    CNN simple, from scratch, optimisé pour un CPU 2 cœurs sans GPU :

    - Conv2D / MaxPooling2D : comme avant, extraction de features.
    - GlobalAveragePooling2D() : AU LIEU DE Flatten(). Flatten garderait les
      20736 valeurs spatiales (18x18x64) de la dernière carte de features,
      ce qui ferait exploser le nombre de paramètres de la couche Dense qui
      suit (20736 x 128 = 2,6 millions de paramètres à lui seul !).
      GlobalAveragePooling2D fait la moyenne de chaque canal sur toute la
      surface -> ne garde qu'UNE valeur par filtre (donc 64 valeurs au lieu
      de 20736). C'est la technique standard des architectures modernes
      (MobileNet, ResNet) pour rester légères. Perte de précision minime,
      parfois même un gain car ça réduit le sur-apprentissage.
    - Dense(64 -> 32) -> Dropout -> Dense(1, sigmoid) : couche de décision,
      maintenant avec très peu de paramètres puisque l'entrée n'est que 64.
    """
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),  # 3 = canaux RGB

        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),  # <- remplace Flatten()
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def run():
    train_ds, val_ds, test_ds = load_datasets()

    model = build_simple_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",  # standard pour une classification à 2 classes
        metrics=["accuracy"],
    )
    model.summary()  # affiche l'architecture + le nombre de paramètres par couche

    # Arrête l'entraînement si la performance sur la validation ne s'améliore
    # plus pendant 3 epochs -> évite de perdre du temps / overfitter.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\n[RESULTAT FINAL] Test loss={test_loss:.4f} | Test accuracy={test_acc:.4f}")

    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    run()
