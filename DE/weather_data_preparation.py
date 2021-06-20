# Databricks notebook source
from pyspark.sql.functions import trim 

# COMMAND ----------

# MAGIC %fs ls /FileStore/data/

# COMMAND ----------

raw_df=spark.read.csv('dbfs:/FileStore/data/weather-raw-chicago-hourly.csv', inferSchema=True,header=True).cache()
raw_df.createOrReplaceTempView("RAW_DF")

# COMMAND ----------

x=spark.sql("""select STATION,DATE,
DailyPrecipitation as PREP,
DailySnowfall as SNOW,
DailySnowDepth as SNWD,
DailyMinimumDryBulbTemperature as TMIN,
DailyMaximumDryBulbTemperature as TMAX
from RAW_DF
where TRIM(REPORT_TYPE2)='SOD'
order by DATE""")
display(x)


# COMMAND ----------

x.repartition(1).write.csv(path="dbfs:/FileStore/data/weather_filtered_chicago/",header=True)

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/data/weather_filtered_chicago/

# COMMAND ----------

y=spark.read.csv("dbfs:/FileStore/data/weather_filtered_chicago/",header=True)
display(y)

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/data/
