# Databricks notebook source
# MAGIC %md
# MAGIC Source : https://github.com/frenchlam/Airline_ontime_prediction/blob/master/Airline_prediction.scala

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/airlines

# COMMAND ----------

# MAGIC %sh
# MAGIC head -50 /dbfs/databricks-datasets/airlines/part-00000

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs

# COMMAND ----------

# MAGIC %sh
# MAGIC tail -50 /dbfs/databricks-datasets/weather/high

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
    raise Exception("get_hour_base10(t) exception : t length should be 3 or 4")
    
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
flights_data_full= "dbfs:/databricks-datasets/airlines/part-000{0[1-9],1[0-5]}"

fl_first_df=spark.read.format("csv").\
option("delimiter",",").\
option("inferschema",True).\
option("header",True).\
load(flights_data_wHead)

fl_remaining_df = spark.read.\
option("header", "false").\
schema(fl_schema).\
csv(flights_data_full)

flightsRaw_df = (fl_first.union(fl_first_df)
                         .fillna("NA")
                         .filter("Origin = 'ORD'" ).cache())

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
# MAGIC cleanup and enrich flights table with label

# COMMAND ----------

sql_statement = """SELECT cast(Year as int) Year, 
cast(Month as int) Month, 
cast(DayofMonth as int) Day, 
cast(DayOfWeek as int) DayOfWeek, 
toDateString(Year, Month, DayofMonth) DateString, 
get_hour_base10(CRSDepTime) decimal_DepTime, 
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

# MAGIC %md
# MAGIC <h2> Weather data definition </h2>

# COMMAND ----------

weather_data_wHead = "dbfs:/databricks-datasets/airlines/part-00000"
weather_data_full= "dbfs:/databricks-datasets/airlines/part-000{0[1-9],1[0-5]}"

fl_first_df=spark.read.format("csv").\
option("delimiter",",").\
option("inferschema",True).\
option("header",True).\
load(flights_data_wHead)

fl_remaining_df = spark.read.\
option("header", "false").\
schema(fl_schema).\
csv(flights_data_full)

flightsRaw_df = (fl_first.union(fl_first_df)
                         .fillna("NA")
                         .filter("Origin = 'ORD'" ).cache())

flightsRaw_df.createOrReplaceTempView("flights_raw")

display(flightsRaw_df)

# COMMAND ----------

# Full Data ... only for use with a cluster on steroids
#weather_data = "dbfs:/databricks-datasets/airlines"
weather_data = "dbfs:/databricks-datasets/weather/part-0000[{0-9}]"

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/weather/

# COMMAND ----------

# MAGIC %sh head -10 /dbfs/databricks-datasets/weather/high_temps

# COMMAND ----------

from pyspark.sql.types import * 

# COMMAND ----------

weatherSchema=StructType(
[
  StructField("station", StringType(), True),
  StructField("DateString", StringType(), True),
  StructField("metric", StringType(), True),
  StructField("value", IntegerType(), True),
  StructField("t1", StringType(), True),
  StructField("t2", StringType(), True),
  StructField("t3", StringType(), True),
  StructField("time", StringType(), True)
])

# COMMAND ----------

weathterRaw_df = (spark.read.format("csv")
                  .option("delimiter",",")
                  .option("quote","")
                  .option("header", "True")
                  .option("inferSchema",True)
                  #.schema(weatherSchema)
                  .load(weather_data))
                  #.filter("station = 'USW00094846'" ))

weathterRaw_df.createOrReplaceTempView("weather_raw")

display(weathterRaw_df)
