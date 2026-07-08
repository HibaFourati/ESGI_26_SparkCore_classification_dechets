"""
config.py
---------
Configuration centrale : chemins, SparkSession, hyperparamètres.
Tous les autres scripts importent ce fichier, rien n'est dupliqué ailleurs.
"""

import os
from pyspark.sql import SparkSession

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# CHEMINS - dataset brut
# ------------------------------------------------------------------
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "data", "DATASET", "TRAIN")
RAW_TEST_DIR = os.path.join(BASE_DIR, "data", "DATASET", "TEST")

# ------------------------------------------------------------------
# CHEMINS - sorties du pipeline (mono-bloc, 3 stages)
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

STAGE1_CLEAN_TRAIN = os.path.join(OUTPUT_DIR, "stage1_clean", "train")
STAGE1_CLEAN_TEST = os.path.join(OUTPUT_DIR, "stage1_clean", "test")
STAGE2_AUGMENTED_TRAIN = os.path.join(OUTPUT_DIR, "stage2_augmented", "train")
STAGE3_READY_TRAIN = os.path.join(OUTPUT_DIR, "stage3_ready", "train")
STAGE3_READY_TEST = os.path.join(OUTPUT_DIR, "stage3_ready", "test")

REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "waste_classifier.keras")

# ------------------------------------------------------------------
# PARAMETRES IMAGES
# ------------------------------------------------------------------
IMG_SIZE = (160, 160)
MIN_VALID_SIZE = (10, 10)
CLASS_MAP = {"O": "organic", "R": "recyclable"}
LABELS = sorted(CLASS_MAP.values())  # ["organic", "recyclable"] -> ordre alphabétique = ordre Keras

# ------------------------------------------------------------------
# ECHANTILLONNAGE (dataset complet trop volumineux pour ce projet)
# ------------------------------------------------------------------
SAMPLE_PER_CLASS_TRAIN = 1000
SAMPLE_PER_CLASS_TEST = 200
SAMPLE_SEED = 42

# ------------------------------------------------------------------
# HYPERPARAMETRES MODELE
# ------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4


def get_spark_session(app_name: str = "WasteClassificationPipeline") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEM", "4g"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
