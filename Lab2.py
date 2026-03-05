from pyspark.sql import SparkSession
from pyspark.sql import functions as FUNC

spark = SparkSession.builder \
    .appName("ProcessStudentData") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")

dataset = spark.read.option("header", "true").csv("dataset/ultimate_student_productivity_dataset_5000.csv")

data = dataset.rdd



