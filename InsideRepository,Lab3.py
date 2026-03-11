from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Topmost Delayed Departure by Trip Count") \
    .master("local[*]") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Preparing the Data: Loading, selecting
trips_df = spark.read.csv("dataset/departuredelays.csv", header=True, inferSchema=True)
trips_select = trips_df.select("origin", "destination", "delay", "distance")

# Processing the Data: Grouping, aggregating, and ordering(by the aggregate trips and the average delay for each)
trips_group = trips_select.groupBy("origin", "destination", "distance").agg(F.round(F.avg("delay"), 2).alias("avg delay"), F.count("*").alias("total trips"))
trips_order = trips_group.orderBy(F.col("total trips").desc(), F.col("avg delay").desc())

# Displaying the Data: Limiting the results
top_delayed = trips_order.limit(20)
top_delayed.show(top_delayed.count(), truncate=False)

# Writing to Text: Formatting, Concatenation and Writing/Overwrite
result = top_delayed.select(F.col("distance").cast("string"))
result_concat = result.select(F.concat_ws(",", *result.columns).alias("one_col"))
result_concat.write.mode("overwrite").text("results.txt")

spark.stop()