# Databricks notebook source
# MAGIC %fs ls dbfs:/databricks-datasets/weather/h

# COMMAND ----------

# MAGIC %sh
# MAGIC tail -50 /dbfs/databricks-datasets/weather/high_temps

# COMMAND ----------

x=spark.read.csv("dbfs:/databricks-datasets/flights/departuredelays.csv", header=True, sep=',')

# COMMAND ----------

# MAGIC %md
# MAGIC From Matthieu:

# COMMAND ----------

initial_df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("dbfs://airlines/part-00000"))
df_schema = initial_df.schema

remaining_df = (spark.read
      .option("header", "false")
      .schema(df_schema)
      .csv("dbfs://airlines/part-000{0[1-9],1[0-5]}")

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/airlines

# COMMAND ----------

flights_data = "dbfs:/databricks-datasets/airlines"
# ou celui ci : dbfs:/databricks-datasets/asa/airlines
weather_data = "dbfs:/databricks-datasets/airlines"

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,DoubleType,IntegerType

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

flightsRaw_df = spark.read.format("csv").\
option("delimiter",",").\
option("quote","").\
option("header", "true").\
option("nullValue", "NA").\
load(flights_data).\
filter("Origin = 'ORD'" ).cache()

flightsRaw_df.createOrReplaceTempView("flights_raw")

display(flightsRaw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC null control

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
