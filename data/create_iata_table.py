# Databricks notebook source
from pyspark.sql.functions import when, col, split

iata=spark.read.json("dbfs:/FileStore/data/airport_codes.json").withColumn("region",split("iso_region", '-')[1])
iata.createOrReplaceTempView("IATA")

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,origin,concat(Dest,' - ',name) as name,sum(delay)
# MAGIC from meetupdb.flights_silver as f join iata as i on i.iata_code=f.dest
# MAGIC where year in (2006,2007,2008) and origin='ORD'
# MAGIC group by year,origin,name, dest
# MAGIC having sum(delay)>100000
# MAGIC order by 1,4;
