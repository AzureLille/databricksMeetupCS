# Databricks notebook source
# MAGIC %sql
# MAGIC select * from meetupdb.flights_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,origin,dest,sum(delay)
# MAGIC from meetupdb.flights_delta
# MAGIC where year=2007
# MAGIC group by year,origin,dest
# MAGIC order by 1,4;

# COMMAND ----------

from pyspark.sql.functions import when, col, split

iata=spark.read.json("dbfs:/FileStore/data/airport_codes.json").withColumn("region",split("iso_region", '-')[1])
iata.createOrReplaceTempView("IATA")

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,origin,name,sum(delay)
# MAGIC from meetupdb.flights_delta as f join iata as i on i.iata_code=f.dest
# MAGIC where year in (2006,2007,2008)
# MAGIC group by year,origin,name
# MAGIC having sum(delay)>50000
# MAGIC order by 1,4;
