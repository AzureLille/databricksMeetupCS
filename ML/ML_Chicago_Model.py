# Databricks notebook source
# Let's take a look at the data
import pandas as ps

train_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas()
X_train = train_pd_df.drop(['label'],axis = 1) \
            .astype({'decimal_DepTime': 'float32'}) \
            .astype({'PREP' : 'float32'}) \
            .astype({'SNOW' : 'float32'}) \
            .astype({'SNWD' : 'float32'})
y_train = train_pd_df["label"]
del train_pd_df #remove original object form memory

test_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas()
X_test = test_pd_df.drop(['label'],axis = 1) \
            .astype({'decimal_DepTime': 'float32'}) \
            .astype({'PREP' : 'float32'}) \
            .astype({'SNOW' : 'float32'}) \
            .astype({'SNWD' : 'float32'})
y_test = test_pd_df["label"]
del test_pd_df #remove original object form memory

display(X_train)


# COMMAND ----------

#sanity check
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)


# COMMAND ----------

X_train.dtypes

# COMMAND ----------

#data preparation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#using ordinal encoder -> compatible with random forest

categorical_ordinal = ['TailNum','Origin','Dest', 'line']
categorical_oneHot = ['UniqueCarrier']
categorical_trf_ordinal = OrdinalEncoder()
categorical_trf_onHot = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[('cat_onHot', categorical_trf_onHot, categorical_oneHot),
                  ('cat_onOrd', categorical_trf_ordinal, categorical_trf_ordinal)])

# COMMAND ----------

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier(n_estimators=100, max_depth = 10, n_jobs = -1))
                     ])


clf.fit(X_train, y_train)

# COMMAND ----------




# COMMAND ----------

# MAGIC %pip show scikit-learn

# COMMAND ----------


