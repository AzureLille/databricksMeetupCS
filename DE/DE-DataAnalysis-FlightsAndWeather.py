# Databricks notebook source
# MAGIC %sql
# MAGIC select * from meetupdb.flights_delta;

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,origin,dest,sum(delay)
# MAGIC from meetupdb.flights_delta
# MAGIC where year=2007
# MAGIC group by year,origin,dest
# MAGIC having sum(delay)>100000
# MAGIC order by 1,4;

# COMMAND ----------

from pyspark.sql.functions import when, col, split

iata=spark.read.json("dbfs:/FileStore/data/airport_codes.json").withColumn("region",split("iso_region", '-')[1])
iata.createOrReplaceTempView("IATA")

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,origin,concat(Dest,' - ',name) as name,sum(delay)
# MAGIC from meetupdb.flights_delta as f join iata as i on i.iata_code=f.dest
# MAGIC where year in (2006,2007,2008)
# MAGIC group by year,origin,name, dest
# MAGIC having sum(delay)>100000
# MAGIC order by 1,4;

# COMMAND ----------

# MAGIC %sql
# MAGIC select year,dest,sum(delay)
# MAGIC from meetupdb.flights_delta
# MAGIC where dest in ('EWR','LGA')
# MAGIC group by year,dest
# MAGIC order by year;

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view FD
# MAGIC as select year,count(1) as delayed,sum(delay) as delays_in_seconds
# MAGIC from  meetupdb.flights_delta
# MAGIC where delay>0
# MAGIC group by year
# MAGIC order by 1;
# MAGIC 
# MAGIC select * from fd;

# COMMAND ----------

myfd=spark.read.table("FD").toPandas()
print(myfd.columns)

# COMMAND ----------

import plotly.graph_objects as go
from plotly.graph_objects import Scatter

fig = go.Figure()
fig.add_trace(go.Scatter(x=myfd.year, y=myfd.delayed, name="Delayed flights", line_color="blue"))
fig.add_trace(go.Scatter(x=myfd.year, y=myfd.delays_in_seconds, name="Delays in seconds", line_color="red", yaxis="y2"))

fig.update_xaxes(tickangle=45,
                 tickmode = 'array',
                 tickvals = myfd['year'],
                 ticktext= myfd['year'])

fig.update_layout(
  xaxis=dict(
    title="Year"
  ),
  yaxis=dict(
        title="<b>delayed flights</b>",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightblue'
  ),
  yaxis2=dict(
        title="<b>delays in seconds</b>",
         anchor="x",
        overlaying="y",
        side="right",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightpink'
    )
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## More charts and examples here:
# MAGIC <ul>
# MAGIC   <li> In Python : https://docs.databricks.com/_static/notebooks/charts-and-graphs-python.html </li>
# MAGIC   <li> In Scala  : https://docs.databricks.com/_static/notebooks/charts-and-graphs-scala.html </li>
# MAGIC   </ul>
