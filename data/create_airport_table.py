# Databricks notebook source
storage_account_name = "***"
storage_account_access_key = "***"

# COMMAND ----------

dbutils.fs.mount(
  source = f"wasbs://dataref@{storage_account_name}.blob.core.windows.net",
  mount_point = "/mnt/databricks-meetup-ref",
  extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net":storage_account_access_key})

# COMMAND ----------

# MAGIC %md
# MAGIC Read the data

# COMMAND ----------

df = spark.read.format('csv').option("inferSchema", "true").option("header", "true").load('/mnt/databricks-meetup-ref/airport_ref.csv')
display(df)

# COMMAND ----------

df.createOrReplaceTempView("tmp_airport_referential")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   tmp_airport_referential

# COMMAND ----------

df.write.format("parquet").saveAsTable("airport_referential")
