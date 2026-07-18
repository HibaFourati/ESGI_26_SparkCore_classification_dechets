import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    get_spark_session,
    STAGE2_AUGMENTED_TRAIN,
    STAGE3_READY_TRAIN,
    STAGE1_CLEAN_TEST,
    STAGE3_READY_TEST,
)

from spark_pipeline import (
    stage_2_augmentation,
    stage_3_export,
)

spark = get_spark_session("WasteStages2And3")

try:
    stage_2_augmentation(spark)

    stage_3_export(
        spark,
        STAGE2_AUGMENTED_TRAIN,
        STAGE3_READY_TRAIN,
        "train",
    )

    stage_3_export(
        spark,
        STAGE1_CLEAN_TEST,
        STAGE3_READY_TEST,
        "test",
    )

    print("\n[STAGES 2 ET 3] Terminés avec succès.")

finally:
    spark.stop()
