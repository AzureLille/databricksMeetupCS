# Databricks notebook source
# Let's take a look at the data
import pandas as pd

train_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas().sample(frac=0.3, replace=True, random_state=1)
X_train = train_pd_df.drop(['label'],axis = 1)
X_train['decimal_DepTime'] = X_train['decimal_DepTime'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_train['PREP'] = X_train['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree
X_train['SNOW'] = X_train['PREP'].apply(lambda x: int(x*100)).astype('int32') #Need discreet values for decision tree

y_train = train_pd_df["label"]
del train_pd_df #remove original object form memory

test_pd_df = spark.sql('''SELECT *, concat(Origin, Dest) as line
                        FROM meetupdb.flights_gold_ml 
                        WHERE Year = 2007
                        ''')\
                    .drop("DateString","DATE","STATION")\
                    .toPandas()
X_test = test_pd_df.drop(['label'],axis = 1)
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

X_train.dtypes

# COMMAND ----------

#data preparation
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#using ordinal encoder -> compatible with random forest

categorical_ordinal = ['TailNum','Origin','Dest', 'line', 'UniqueCarrier' ]
#categorical_oneHot = ['UniqueCarrier']
ordinal_enc = OrdinalEncoder()
onehot_enc = OneHotEncoder()

#ordinal_enc.fit(X_train[categorical_ordinal])
#ordinal_enc.transform(X_train[categorical_ordinal])


# preprocessor = ColumnTransformer(
#     transformers=[
#         ('ord', ordinal_enc, categorical_ordinal),
#         ('onehot', onehot_enc, categorical_oneHot)])

preprocessor = ColumnTransformer(
    transformers=[('ord', ordinal_enc, categorical_ordinal)])

X_train_tr = preprocessor.fit_transform(X_train)
#preprocessor = ColumnTransformer()
#transformers=[('cat_onHot', categorical_trf_onHot, categorical_oneHot),('cat_onOrd', categorical_trf_ordinal, categorical_trf_ordinal)])

X_test_tr = preprocessor.fit_transform(X_test)


# COMMAND ----------

import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model_randomF = RandomForestClassifier( n_jobs = -1)
#model_randomF.fit(X_train_tr, y_train )

# COMMAND ----------

from sklearn.metrics import precision_recall_fscore_support
y_pred = clf.predict(X_test_tr)
confusion_matrix = pd.DataFrame(precision_recall_fscore_support(y_test, y_pred, average='macro'))

# COMMAND ----------

Display()

# COMMAND ----------



# COMMAND ----------

from sklearn.metrics import roc_curve, auc                # to import roc curve abd auc metrics for evaluation 
#from sklearn.grid_search import GridSearchCV              # grid search is used for hyperparameters-optimization
import seaborn as sns                                     # Python graphing library
import matplotlib.pyplot as plt

def Performance(Model,Y,X):
    # Perforamnce of the model
    fpr, tpr, _ = roc_curve(Y, Model.predict_proba(X)[:,1])
    AUC  = auc(fpr, tpr)
    print ('the AUC is : %0.4f' %  AUC)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

params = { "n_estimators"      : [100,150,200],
           "max_depth"         : [3, 10],
           "min_samples_split" : [2, 10] }
grid_search = GridSearchCV(model_randomF, params)

# COMMAND ----------

grid_search.fit(X_train_tr, y_train)

# COMMAND ----------

Performance(Model=clf,Y=y_test,X=X_test_tr)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
search_space = {
     'n_estimators': hp.choice('n_estimators', [100,150,200]),
     'max_depth': hp.choice('max_depth', [5, 10, 20]),
     'min_samples_split': hp.choice("min_samples_split", [5 , 10, 15])
     }


from sklearn.model_selection import cross_val_score
def acc_model(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X_train_tr, y_train).mean()

#best = 0
def objective(params):
    #global best
    acc = acc_model(params)
    #if acc > best:
    #    best = acc
    print ('new best:', best, params)
    return {'loss': -acc, 'status': STATUS_OK}



# COMMAND ----------

from hyperopt import SparkTrials
algo=tpe.suggest
spark_trials = SparkTrials(parallelism=12)

best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=algo,
    max_evals=20,
    trials=spark_trials)


# COMMAND ----------


