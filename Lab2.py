from pyspark.sql import SparkSession
from pyspark.sql import functions as FUNC

spark = SparkSession.builder \
    .appName("ProcessStudentData") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("ERROR")

dataset = spark.read.csv("dataset/ultimate_student_productivity_dataset_5000.csv", header=True, inferSchema=True)

# Repartitioning the dataset by "student_id" into 5 partitions [hash partitioning]
dataPart1 = dataset.repartitionByRange(5, "student_id")



dataTransform1 = dataPart1.withColumn("age_group", FUNC.when(FUNC.col("age") < 20, "Teen").when((FUNC.col("age") >= 20) & (FUNC.col("age") < 24), "Young Adult").otherwise("Adult"))

group_average = dataTransform1.groupBy("age_group").agg(FUNC.avg("sleep_hours").alias("avg_slp_hrs"), FUNC.avg("social_media_hours").alias("avg_soc_med_hrs"), FUNC.avg("burnout_level").alias("avg_burnout_lvl"))

dataWithAvgs = dataTransform1.join(group_average, on="age_group")

#assigning true or false values [1, 0] to be summed up later
dataIndicators = (dataWithAvgs.withColumn("Below Average Sleep Hrs Count", FUNC.when(FUNC.col("sleep_hours") < FUNC.col("avg_slp_hrs"), 1).otherwise(0))
.withColumn("Above Average Social Media Hrs Count", FUNC.when(FUNC.col("social_media_hours") > FUNC.col("avg_soc_med_hrs"), 1).otherwise(0))
.withColumn("Above Average Burnout Level Count", FUNC.when(FUNC.col("burnout_level") > FUNC.col("avg_burnout_lvl"), 1).otherwise(0)))

#Summing up the assigned values for each age group
aggregatedData = dataIndicators.groupBy("age_group").agg(FUNC.sum("Below Average Sleep Hrs Count").alias("Total Below Avg Sleep Hrs"), FUNC.sum("Above Average Social Media Hrs Count").alias("Total Above Avg Social Media Hrs"), FUNC.sum("Above Average Burnout Level Count").alias("Total Above Avg Burnout Level"),
FUNC.count("*").alias("Total Population"))

#Getting the percentage for each summation relative to the total population for each age group
percentData = aggregatedData.withColumn("% Below Avg Sleep Hrds", (FUNC.col("Total Below Avg Sleep Hrs") / FUNC.col("Total Population")) * 100).withColumn("% Above Avg Social Media Hrs", (FUNC.col("Total Above Avg Social Media Hrs") / FUNC.col("Total Population")) * 100).withColumn("% Above Avg Burnout Level", (FUNC.col("Total Above Avg Burnout Level") / FUNC.col("Total Population")) * 100)

#Removing the long decimal places and adding the percent symbol
cleanData = percentData.select("age_group", "Total Population", FUNC.concat(FUNC.format_number("% Below Avg Sleep Hrds", 2), FUNC.lit("%")).alias("% Below Avg Sleep Hrs"), FUNC.concat(FUNC.format_number("% Above Avg Social Media Hrs", 2), FUNC.lit("%")).alias("% Above Avg Social Media Hrs"), FUNC.concat(FUNC.format_number("% Above Avg Burnout Level", 2), FUNC.lit("%")).alias("% Above Avg Burnout Level"))




#Repartitioning the dataset by the total population into 1 partition.
display = cleanData.repartitionByRange(1, "Total Population")

display.show()

spark.stop()








