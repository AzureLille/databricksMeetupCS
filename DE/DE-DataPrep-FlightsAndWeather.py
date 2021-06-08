# Databricks notebook source
with open("/dbfs/databricks-datasets/airlines/README.md") as f:
  print(f.read())

# COMMAND ----------

# MAGIC %md
# MAGIC Inspired by : https://github.com/frenchlam/Airline_ontime_prediction/blob/master/Airline_prediction.scala

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>Fun starts here !</h1>

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,DoubleType,IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>UDF Defs</h2>

# COMMAND ----------

def toDateString(year: str, month: str, day: str) -> str :
  Y = '{0:04d}'.format(int(year))
  M = '{0:02d}'.format(int(month))
  D = '{0:02d}'.format(int(day))
  return Y+M+D
spark.udf.register("toDateString",toDateString,StringType())

def get_hour_base10(t: str) -> float :
  temp_time=t
  le=len(t)
  
  if (le<3 or le>4):
    raise Exception("temp_time "+temp_time+"get_hour_base10(t) exception : t length should be 3 or 4")
    
  if(le==3):
    temp_time=str(0)+t
  
  hour=int(temp_time[0:2])
  if (hour<0 or hour>=24):
    raise Exception("get_hour_base10(t) exception : hour should be between 0 and 23")
    
  minute=int(temp_time[2:4])
  if (minute<0 or minute>=60):
    raise Exception("get_hour_base10(t) exception : minute should be between 0 and 59")
  
  dec_time=hour+(minute/60)
  return dec_time
spark.udf.register("get_hour_base10",get_hour_base10,DoubleType())

def delay_label(delay: str) -> int:
  label=0
  
  if (delay is not None):
    delayInt = int(delay)
    if (delayInt > 15):
      label=1
      
  return label
spark.udf.register("delay_label",delay_label,IntegerType())

# COMMAND ----------

# MAGIC %md
# MAGIC <h2> Flights data definition </h2>
# MAGIC Files are in CSV format but in multipart with header defined in the first one : dbfs:/databricks-datasets/airlines/part-00000
# MAGIC So we built our flightsRaw Dataframe in three steps :
# MAGIC <ol>
# MAGIC   <li> Reading the first file in order to get the schema</li>
# MAGIC   <li> Reading the rest of files</li>
# MAGIC   <li> Merge both dataframes (with union), filtering, fill null values with "NA" and the caching it for future use
# MAGIC </ol>

# COMMAND ----------

flights_data_wHead = "dbfs:/databricks-datasets/airlines/part-00000"
# With a single node cluster ... Play this
flights_data_full= "dbfs:/databricks-datasets/airlines/part-009[0-5][0-9]" ## Here we got date from 2003 to 2008 (and a bunch of older ones)

# With a multinode cluster ... Play this (Better run on 4 nodes)
#flights_data_full= "dbfs:/databricks-datasets/airlines/part-00[5-9][0-9][0-9]" ## Here we got date from 2003 to 2008 (and a bunch of older ones)

fl_first_df= (spark.read.format("csv")
              .option("delimiter",",")
              .option("inferschema",True)
              .option("header",True)
              .option("nullValue", "NA")
              .load(flights_data_wHead)
             )

fl_schema=fl_first_df.schema

fl_remaining_df = (spark.read
                   .option("header", "false")
                   .option("nullValue", "NA")
                   .schema(fl_schema)
                   .csv(flights_data_full)
                  )

flightsRaw_df = (fl_first_df.union(fl_remaining_df)
                 .filter("Origin = 'ORD'" )
                 .cache()
                )

flightsRaw_df.createOrReplaceTempView("flights_raw")

display(flightsRaw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Quality Control</h3> 

# COMMAND ----------

from pyspark.sql.functions import count,when,isnan
display(flightsRaw_df.select([count(when(isnan(c), c)).alias(c) for c in flightsRaw_df.columns]))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Cleanup and enrich flights view with label</h3>

# COMMAND ----------

sql_statement = """SELECT cast(Year as int) Year, 
                   cast(Month as int) Month, 
                   cast(DayofMonth as int) Day, 
                   cast(DayOfWeek as int) DayOfWeek, 
                   toDateString(Year, Month, DayofMonth) DateString, 
                   get_hour_base10(lpad(cast(CRSDepTime as string),4,'0')) decimal_DepTime, 
                   UniqueCarrier,  
                   cast(FlightNum as int) FlightNum,  
                   IFNULL(TailNum, 'N/A') AS TailNum, 
                   Origin ,  
                   Dest , 
                   cast(Distance as int) Distance, 
                   IFNULL(cast(DepDelay as int ), 0) delay, 
                   delay_label(DepDelay) label  
                   FROM flights_raw """

flights_df = spark.sql(sql_statement).cache()

flights_df.createOrReplaceTempView("flights")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from flights;

# COMMAND ----------

# MAGIC %md
# MAGIC <h2> Weather data definition </h2>

# COMMAND ----------

# Full Data ... only for use with a cluster on steroids
#weather_data = "dbfs:/databricks-datasets/airlines"
weather_data = "dbfs:/FileStore/data/weather_filtered_chicago/"

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/data/weather_filtered_chicago/

# COMMAND ----------

weather_raw_df=spark.read.csv("dbfs:/FileStore/data/weather_filtered_chicago/",header=True,inferSchema=True)
weather_raw_df.createOrReplaceTempView("WEATHER_RAW")

weather_df=spark.sql("""
                     SELECT *,toDateString(YEAR, MONTH, DAY) DateString 
                     FROM
                     (
                       SELECT extract(year from date)  as YEAR,
                       extract(month from date) as MONTH,
                       extract(day from date) as DAY,
                       *
                       from WEATHER_RAW
                     )
""")
display(weather_df)

# COMMAND ----------

joinedDF = (flights_df.join(weather_df, flights_df.DateString == weather_df.DateString)).drop("DateString","YEAR","MONTH","DAY")
          
display(joinedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> Now creating a global table in order to reuse it in another context (notebook, app, SQL Databricks etc.)</h3>

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS meetupdb;
# MAGIC 
# MAGIC DROP TABLE IF EXISTS meetupdb.flight_and_weather_parquet;
# MAGIC DROP TABLE IF EXISTS meetupdb.flight_and_weather_delta;
# MAGIC 
# MAGIC DROP TABLE IF EXISTS meetupdb.flights_parquet;
# MAGIC DROP TABLE IF EXISTS meetupdb.flights_delta;

# COMMAND ----------

joinedDF.write.saveAsTable("meetupdb.flight_and_weather_parquet", format="parquet")
joinedDF.write.saveAsTable("meetupdb.flight_and_weather_delta", format="delta")

flights_df.write.saveAsTable("meetupdb.flights_parquet",format="parquet")
flights_df.write.saveAsTable("meetupdb.flights_delta",format="delta")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from meetupdb.flights_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from meetupdb.flights_delta;

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO
# MAGIC Préparer un job Cluster (filtre sur l'année en widget)
# MAGIC Plus blabla sur génération de package, wheel etc.
