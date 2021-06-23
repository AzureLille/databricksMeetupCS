# Databricks notebook source
# MAGIC %sql 
# MAGIC SELECT * FROM meetupdb.flights_gold_ml 
# MAGIC limit 100

# COMMAND ----------

# MAGIC %md 
# MAGIC ## using Pandas 
# MAGIC With a background in Python dev and DataScience I'm more familiar with Pandas so let's use that

# COMMAND ----------

# DBTITLE 1,Test and Train datasets - using 2007 data to predict 2008 flights 
import pandas as pd

# read data from Spark 
train_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas().sample(frac=0.3, replace=True, random_state=1)

test_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas()
X_test = test_pd_df.drop(['label'],axis = 1)

# COMMAND ----------

#get Delta version - to cpature with MLFlow 
delta_version = sql("SELECT MAX(version) AS VERSION FROM (DESCRIBE HISTORY meetupdb.flights_gold_ml)").head()[0]

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Feature Engineering and preparation
# MAGIC Dicretisation of value and Categorical value encoding 

# COMMAND ----------

# DBTITLE 1,Discretisation of features
X_train = train_pd_df.drop(['label'],axis = 1)
X_train['decimal_DepTime'] = X_train['decimal_DepTime'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_train['PREP'] = X_train['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_train['SNOW'] = X_train['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree

y_train = train_pd_df["label"]
del train_pd_df #remove original object form memory


X_test['decimal_DepTime'] = X_test['decimal_DepTime'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_test['PREP'] = X_test['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_test['SNOW'] = X_test['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree

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

# DBTITLE 1,Categorical feature encoding
#data preparation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

#using ordinal encoder -> compatible with random forest
categorical_ordinal = ['TailNum','Origin','Dest', 'line', 'UniqueCarrier' ]
ordinal_enc = OrdinalEncoder()

preprocessor = ColumnTransformer(
    transformers=[('ord', ordinal_enc, categorical_ordinal)])

X_train_tr = preprocessor.fit_transform(X_train)
X_test_tr = preprocessor.fit_transform(X_test)

X_train_tr.shape

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Training 
# MAGIC Using Random Forest - just because... 

# COMMAND ----------

# DBTITLE 1,Baseline
import mlflow
from sklearn.ensemble import RandomForestClassifier

params_init = { "random_state" : 0, "max_depth" : 5 , "min_samples_split" : 2 }
model_baseline = RandomForestClassifier( **params_init )

mlflow.sklearn.autolog()
with mlflow.start_run(run_name = "Sckit Learn unit" ) as run:
  model_baseline.fit(X_train_tr, y_train)
  mlflow.log_param("delta_version", delta_version) # Log Delta table version along with the model
  mlflow.set_tag("project", "Airline Ontime")  # add a tag to the model 

print("model accuracy Score (test dataset) : {}".format(model_baseline.score(X_test_tr, y_test )))
# log parameter -> Not needed with sklearn autolog 
#   for key value in params_init.items : 
#     mlflow.log_param(key, value)
#     mlflow.log_metric("accuracy", model_randomF.score(X_test_tr, y_test ))
  

# COMMAND ----------

# MAGIC %md 
# MAGIC let's take a look at the performance of the baseline model

# COMMAND ----------

from sklearn.metrics import roc_curve, auc                # to import roc curve abd auc metrics for evaluation 
import seaborn as sns                                     # Python graphing library
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve

def rocCurve(Model,Y,X):  
  plot_roc_curve(Model, X, Y)
  plt.show()

def confusionMatrix(Model, Y, X) :
  cm = confusion_matrix(Y, Model.predict(X) , normalize='all')
  cmd = ConfusionMatrixDisplay(cm, display_labels=['business','health'])
  cmd.plot()

# COMMAND ----------

confusionMatrix(model_baseline, y_test, X_test_tr )


# COMMAND ----------

rocCurve(model_baseline, y_test, X_test_tr )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter optimisation 
# MAGIC Let's try to do better with by optimising the params of the model
# MAGIC AKA Grid Search 

# COMMAND ----------

import time
start_time = time.time()

from sklearn.model_selection import GridSearchCV

model_grid_search = RandomForestClassifier()
params = { "n_estimators"      : [100],
           "max_depth"         : [5 , 10],
           "min_samples_split" : [2 , 10] }
grid_search = GridSearchCV(model_grid_search, params, n_jobs = -1)

mlflow.sklearn.autolog()
with mlflow.start_run(run_name = "Sckit Learn unit" ) as run:
  grid_search.fit(X_train_tr, y_train)
  mlflow.log_param("delta_version", delta_version) # Log Delta table version along with the model
  mlflow.set_tag("project", "Airline Ontime")  # add a tag to the model 

  
end_time = time.time()
print("Runtime = {} mins".format(round( (end_time-start_time)/60 ,  2)))

# COMMAND ----------

# MAGIC %md 
# MAGIC Note : MLFLOW SKLearn autolog enregistre que le meilleur run, mais aussi l'ensemble des infos sur les run

# COMMAND ----------

pd.DataFrame(grid_search.cv_results_)

# COMMAND ----------

grid_search.best_score_

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Search best hyper parameters with HyperOpt (Bayesian optimization) accross multiple nodes
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/bayesian-model.png" style="height: 330px"/></div>
# MAGIC 
# MAGIC This model is a good start, but now we want to try multiple hyper-parameter to see how it behaves.
# MAGIC 
# MAGIC GridSearch could be a good way to do it, but not very efficient when the parameter dimension increase and the model is getting slow to train due to a massive amount of data.
# MAGIC 
# MAGIC HyperOpt search accross your parameter space for the minimum loss of your model, using Baysian optimization instead of a random walk

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/hyperopt-spark.png" style="height: 300px; margin-left:20px"/></div>
# MAGIC #### Distribute HyperOpt accross multiple nodes
# MAGIC HyperOpt is ready to be used with your spark cluster and can automatomatically distribute the workload accross multiple instances.
# MAGIC 
# MAGIC Spark Hyperopt also automatically log all trials to MLFLow!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Hyperopt Cost minimisation function
# MAGIC In our case we'll base it on model accuracy   
# MAGIC We'll also define the search space

# COMMAND ----------

# DBTITLE 1,Setup Cost function 
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import STATUS_OK

def objective(params):
    #Note hp.quniform returns a Double -> Need to be converted to int ! 
    model_baseline = RandomForestClassifier( max_depth = int(params['max_depth']) , 
                                             min_samples_split = int(params['min_samples_split'] ), 
                                             criterion =  params['criterion']  )
    accuracy = cross_val_score(model_baseline, X_test_tr, y_test, cv = 2).mean() # cv should be at 5 but speeding things up... 
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK, }


# COMMAND ----------

# Alternative cost function that returns a model
# def objective(params):
#     clf = RandomForestClassifier( **params , n_jobs = -1 )
#     mlflow.sklearn.autolog()
#     with mlflow.start_run(run_name = "hyperopt" ) :
#         clf.fit(X_train_tr, y_train)
#         score = clf.score(X_test_tr, y_test)
#     # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
#     return {'loss': -score, 'status': STATUS_OK, 'model': clf }

# COMMAND ----------

# DBTITLE 1,Setup Search Space
#Search Space 
from hyperopt import hp, STATUS_OK
search_space = {
        'max_depth': hp.quniform('max_depth', 5, 16 , 2 ),
        'min_samples_split' : hp.quniform('min_samples_split', 2, 11, 2 ),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    }
# Note hp.quniform returns a double !!!!!
# see https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions 

# COMMAND ----------

from hyperopt import fmin, tpe, SparkTrials, space_eval
spark_trials = SparkTrials(parallelism = 16 ) 

import time
start_time = time.time()

mlflow.sklearn.autolog()
with mlflow.start_run(run_name = "hyperopt", nested = True ) as run:
  best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=16,
    trials = spark_trials)
  
  #create and register model with best params ( model registration is automatic with sklearn.autolog() - when using fit. )
  best_params = space_eval(search_space,best)
  clf = RandomForestClassifier( max_depth = int(best_params['max_depth']) , 
                                min_samples_split = int(best_params['min_samples_split'] ), 
                                criterion =  best_params['criterion'] , n_jobs = -1 )
  clf.fit(X_train_tr, y_train)
  
  #log extra params 
  mlflow.log_param("delta_version", delta_version)
  mlflow.set_tag("model", "random_forest_scikit")  
  
#   for key, value in best_params.items():
#     mlflow.log_param(key, value)
  #mlflow.log_metric("accuracy", -spark_trials.best_trial['result']['loss'])
  
end_time = time.time()
print("Runtime = {} mins".format(round( (end_time-start_time)/60 ,  2)))

#best parameters
print(space_eval(search_space,best))

# COMMAND ----------

# To do 
# Inference using spark and save to table 

# COMMAND ----------

model_from_registry = mlflow.spark.load_model('models:/airlines_ontime_pred/production')
