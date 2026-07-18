from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("Stage1Results")
    .master("local[1]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n" + "=" * 60)
print("STAGE 1 — PRÉTRAITEMENT ET VALIDATION")
print("=" * 60)

train = spark.read.parquet("output/stage1_clean/train")
test = spark.read.parquet("output/stage1_clean/test")

print("\nTRAIN")
print("-" * 60)
print(f"Nombre total d'images retenues : {train.count()}")
train.groupBy("label").count().orderBy("label").show(truncate=False)

print("\nTEST")
print("-" * 60)
print(f"Nombre total d'images retenues : {test.count()}")
test.groupBy("label").count().orderBy("label").show(truncate=False)

print("\nCARACTÉRISTIQUES")
print("-" * 60)
print("Dimensions finales : 160 × 160 pixels")
print("Canaux             : RGB")
print("Format de sortie   : JPEG stocké en Parquet")
print("Classes            : organic / recyclable")
print("Seed               : 42")

print("\nSCHÉMA DU PARQUET TRAIN")
print("-" * 60)
train.printSchema()

print("=" * 60)

spark.stop()
