# Databricks notebook source
from pyspark.sql.functions import trim 

# COMMAND ----------

# MAGIC %fs ls /FileStore/data

# COMMAND ----------

raw_df=spark.read.csv('dbfs:/FileStore/data/weather-raw-chicago-hourly.csv', inferSchema=True,header=True).cache()
raw_df.createOrReplaceTempView("RAW_DF")

# COMMAND ----------

# ORD = Chicago
ORD_DF=spark.sql("""select STATION,
                  "ORD" as Airport,
                  DATE,
                  extract(year from date)  as YEAR,
                  extract(month from date) as MONTH,
                  extract(day from date) as DAY,
                  date_format(DATE, "yyyyMMdd") as DateString,
                  CAST( 
                      CASE 
                          WHEN DailyPrecipitation = 'T' then '0.0'
                          else DailyPrecipitation
                      END as double 
                      )as PREP,
                  CAST( 
                      CASE 
                          WHEN DailySnowfall = 'T' then '0.0'
                          else DailySnowfall
                      END as double 
                      )as SNOW,
                  CAST( 
                      CASE 
                          WHEN DailySnowDepth = 'T' then '0'
                          else DailySnowfall
                      END as int 
                      )as SNWD,
                  DailyMinimumDryBulbTemperature as TMIN,
                  DailyMaximumDryBulbTemperature as TMAX
              from RAW_DF
              where TRIM(REPORT_TYPE2)='SOD'
              order by DATE""")
display(ORD_DF)


# COMMAND ----------

#SEA = Seattle Tacoma
SEA_DF=spark.sql("""select STATION,
                  "SEA" as Airport,
                  DATE,
                  extract(year from date)  as YEAR,
                  extract(month from date) as MONTH,
                  extract(day from date) as DAY,
                  date_format(DATE, "yyyyMMdd") as DateString,
                  CAST( 
                      CASE 
                          WHEN DailyPrecipitation = 'T' then '0.0'
                          else DailyPrecipitation
                      END as double 
                      )as PREP,
                  CAST( 
                      CASE 
                          WHEN DailySnowfall = 'T' then '0.0'
                          else DailySnowfall
                      END as double 
                      )as SNOW,
                  CAST( 
                      CASE 
                          WHEN DailySnowDepth = 'T' then '0'
                          else DailySnowfall
                      END as int 
                      )as SNWD,
                  DailyMinimumDryBulbTemperature as TMIN,
                  DailyMaximumDryBulbTemperature as TMAX
              from RAW_DF
              where TRIM(REPORT_TYPE2)='SOD'
              order by DATE""")
display(SEA_DF)

# COMMAND ----------

all_airports_df = ORD_DF.union(SEA_DF)
all_airports_df.createOrReplaceTempView("weather_data_df")

# COMMAND ----------

# MAGIC %sql
# MAGIC use meetupdb ;
# MAGIC DROP TABLE IF EXISTS weather_data ;
# MAGIC 
# MAGIC CREATE TABLE weather_data
# MAGIC USING DELTA 
# MAGIC AS (
# MAGIC     SELECT * FROM weather_data_df
# MAGIC     )

# COMMAND ----------

# MAGIC %sql select * from weather_data

# COMMAND ----------

#to_CSV
all_airports_df.repartition(3).write.mode("overwrite").csv(path="dbfs:/FileStore/data/weather_filtered/",header=True)

# COMMAND ----------

y=spark.read.csv("dbfs:/FileStore/data/weather_filtered",header=True)
display(y)

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/data/
