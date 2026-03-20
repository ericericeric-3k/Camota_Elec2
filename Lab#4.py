from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder \
    .appName("Visualizing Data") \
    .master("local[*]") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


student_df = spark.read.csv("datasets/student.csv", header=True, inferSchema=True)
departDelay_df = spark.read.csv("datasets/departuredelays.csv", header=True, inferSchema=True)

#clean
select_student = student_df.select("student_id", "age", "sleep_hours", "social_media_hours", "burnout_level")
select_departDelay = departDelay_df.select("origin", "destination", "delay", "distance")

#PROCESSING STUDENT DATA
part_student = select_student.repartitionByRange(5, "student_id")

age_grouped = part_student.withColumn("age_group", F.when(F.col("age") < 20, "Teen").when((F.col("age") >= 20) & (F.col("age") < 24), "Young Adult").otherwise("Adult"))

group_avg = age_grouped.groupBy("age_group").agg(F.avg("sleep_hours").alias("avg_slp_hrs"), F.avg("social_media_hours").alias("avg_soc_med_hrs"), F.avg("burnout_level").alias("avg_burnout_lvl"))

consolidate_avg = group_avg.join(age_grouped, on="age_group")

indicators = (consolidate_avg.withColumn("Below Average Sleep Hrs Count", F.when(F.col("sleep_hours") < F.col("avg_slp_hrs"), 1).otherwise(0))
.withColumn("Above Average Social Media Hrs Count", F.when(F.col("social_media_hours") > F.col("avg_soc_med_hrs"), 1).otherwise(0))
.withColumn("Above Average Burnout Level Count", F.when(F.col("burnout_level") > F.col("avg_burnout_lvl"), 1).otherwise(0)))

aggregated = indicators.groupBy("age_group").agg(F.sum("Below Average Sleep Hrs Count").alias("Total Below Avg Sleep Hrs"), F.sum("Above Average Social Media Hrs Count").alias("Total Above Avg Social Media Hrs"), F.sum("Above Average Burnout Level Count").alias("Total Above Avg Burnout Level"), F.count("*").alias("Total Population"))

percent = aggregated.withColumn("% Below Avg Sleep Hrs", (F.col("Total Below Avg Sleep Hrs") / F.col("Total Population")) * 100).withColumn("% Above Avg Social Media Hrs", (F.col("Total Above Avg Social Media Hrs") / F.col("Total Population")) * 100).withColumn("% Above Avg Burnout Level", (F.col("Total Above Avg Burnout Level") / F.col("Total Population")) * 100)

clean = percent.select("age_group", "Total Population", F.concat(F.format_number("% Below Avg Sleep Hrs", 2), F.lit("%")).alias("% Below Avg Sleep Hrs"), F.concat(F.format_number("% Above Avg Social Media Hrs", 2), F.lit("%")).alias("% Above Avg Social Media Hrs"), F.concat(F.format_number("% Above Avg Burnout Level", 2), F.lit("%")).alias("% Above Avg Burnout Level"))

display_student = clean.repartitionByRange(1, "Total Population")

display_student.show()


#Visualization_STUDENT1 - Total Population by Age Group
age_group_pd = display_student.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x="age_group", y="Total Population", data=age_group_pd, palette=["yellow", "blue", "green"])
plt.title("Total Population by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Population")
plt.grid()
plt.show()

#Visualization_STUDENT2 - Percentage of Students Below Average Sleep Hours by Age Group
indicators_pd = indicators.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x="age_group", y="Below Average Sleep Hrs Count", data=indicators_pd, palette=["yellow", "blue", "green"])
plt.title("Percentage of Students Below Average Sleep Hours by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Percentage Below Average Sleep Hours")
plt.grid()
plt.show()

#Visualization_STUDENT3 - Percentage of Students Above Average Social Media Hours by Age Group
plt.figure(figsize=(12, 6))
sns.barplot(x="age_group", y="Above Average Social Media Hrs Count", data=indicators_pd, palette=["yellow", "blue", "green"])
plt.title("Percentage of Students Above Average Social Media Hours by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Percentage Above Average Social Media Hours")
plt.grid()
plt.show()

#Visualization_STUDENT4 - Sleep Hours vs Burnout Level
teen_pd = age_grouped.filter(F.col("age_group") == "Teen").toPandas()
yAdult_pd = age_grouped.filter(F.col("age_group") == "Young Adult").toPandas()
adult_pd = age_grouped.filter(F.col("age_group") == "Adult").toPandas()
plt.figure(figsize=(12, 6))
sns.scatterplot(x="sleep_hours", y="burnout_level", data=teen_pd, label="Teen", color="yellow")
sns.scatterplot(x="sleep_hours", y="burnout_level", data=yAdult_pd, label="Young Adult", color="blue")
sns.scatterplot(x="sleep_hours", y="burnout_level", data=adult_pd, label="Adult", color="green")
plt.title("Sleep Hours vs Burnout Level")
plt.xlabel("Sleep Hours")
plt.ylabel("Burnout Level")
plt.legend()
plt.grid()
plt.show()

#Visualization_STUDENT5 - Social Media Hours vs Burnout Level by Age Group
plt.figure(figsize=(12, 6))
sns.scatterplot(x="social_media_hours", y="burnout_level", data=teen_pd, label="Teen", color="yellow")
sns.scatterplot(x="social_media_hours", y="burnout_level", data=yAdult_pd, label="Young Adult", color="blue")
sns.scatterplot(x="social_media_hours", y="burnout_level", data=adult_pd, label="Adult", color="green")
plt.title("Social Media Hours vs Burnout Level by Age Group")
plt.xlabel("Social Media Hours")
plt.ylabel("Burnout Level")
plt.legend()
plt.grid()
plt.show()


#PROCESSING DEPARTURE_DELAY DATA
trips_grouped = select_departDelay.groupBy("origin", "destination", "distance").agg(F.round(F.avg("delay"), 2).alias("avg_delay"), F.count("*").alias("total_trips"))

trips_ordered = trips_grouped.orderBy(F.col("total_trips").desc(), F.col("avg_delay").desc())

top_delayed = trips_ordered.limit(20)

trip_limit = trips_ordered.limit(150)

top_delayed.show(top_delayed.count())

trip_limit_pd = trip_limit.toPandas()


#Visualization_DEPARTDELAY1 - Average Delay vs Total Trips by Distance Categories
trip_Mx_pd = trips_ordered.limit(1000).filter(F.col("distance") >= 784).toPandas()
trip_Mid_pd = trips_ordered.limit(1000).filter((F.col("distance") >= 383) & (F.col("distance") < 784)).toPandas()
trip_Low_pd = trips_ordered.limit(1000).filter(F.col("distance") < 383).toPandas()
plt.figure(figsize=(12, 6))
sns.scatterplot(x="total_trips", y="avg_delay", data=trip_Mx_pd, label="Long Distance (>= 784)", color="red")
sns.scatterplot(x="total_trips", y="avg_delay", data=trip_Mid_pd, label="Medium Distance (383 - 783)", color="orange")
sns.scatterplot(x="total_trips", y="avg_delay", data=trip_Low_pd, label="Short Distance (< 383)", color="green")
plt.title("Average Delay vs Total Trips [Top 1000 Trips]")
plt.xlabel("Total Trips")
plt.ylabel("Average Delay")
plt.grid()
plt.show()

#Visualization_DEPARTDELAY2 - Average Delay for Trips Originating from SFO
trip_filtered = trips_ordered.filter(F.col("origin") == "SFO")
trip_sfo_pd = trip_filtered.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x="destination", y="avg_delay", data=trip_sfo_pd)
plt.title("Average Delay for Trips Originating from SFO")
plt.xlabel("Destination")
plt.ylabel("Average Delay")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#Visualization_DEPARTDELAY3 - Average Delay vs Distance for Trips Originating from SFO
plt.figure(figsize=(12, 6))
sns.lineplot(x="distance", y="avg_delay", data=trip_sfo_pd)
plt.title("Average Delay vs Distance of Trips Originating from SFO")
plt.xlabel("Distance")
plt.ylabel("Average Delay")
plt.grid()
plt.show()

#Visualization_DEPARTDELAY4 - Average Delay for Top 150 Trips by Origin
plt.figure(figsize=(12, 6))
sns.barplot(x="origin", y="avg_delay", data=trip_limit_pd)
plt.title("Average Delay for Top 150 Trips by Origin")
plt.xlabel("Origin")
plt.ylabel("Average Delay")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#Visualization_DEPARTDELAY5 - Distribution of Average Delays for Top 150 Trips
plt.figure(figsize=(12, 6))
sns.histplot(trip_limit_pd["avg_delay"], bins=20, kde=True)
plt.title("Distribution of Average Delays for Top 150 Trips")
plt.xlabel("Average Delay")
plt.ylabel("Frequency")
plt.grid()
plt.show()




spark.stop()




