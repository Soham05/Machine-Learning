
# coding: utf-8

# Soham Bhalerao
# 

# In[46]:

import os
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
   
print(os.environ['SPARK_HOME'])
from pyspark.sql import SparkSession


#following code starts a spark session for you. 
#from this configuration you can control the scalability of your spark application for this notebook
spark = SparkSession     .builder     .appName("Pyspar Rule Set Prediction")     .config("spark.driver.memory","24g")     .config("spark.executor.instances","45",)     .config("spark.yarn.queue","default")     .config("spark.master","yarn")     .config("spark.submit.deployMode","client")    .enableHiveSupport()    .getOrCreate()     
    
    
print(spark.version)


# In[47]:

data = spark.sql('select * from claims_cert_dev.cod_ctop_25k_bp3')
data_cols = data.columns
print(data.count(),len(data.columns))


# ## Null Treatment

# In[1]:

from pyspark.sql.functions import when,col,count,isnan,lit,sum
### % of non-null values: Function
def count_not_null(c,nan_as_null=True):
    """""" 
    ## False -> 0
    ##True -> 1
    """"""
    pred = col(c).isNotNull() & (~isnan(c) if nan_as_null else lit(True))
    return sum(pred.cast("integer").alias(c))


### drop columns with 70% or more null values
def drop_null_columns(data):
    
    nonnull_counts = data.select(*[(count(c) / count("*")).alias(c) for c in data.columns]).collect()[0].asDict()
    to_drop = [k for k, v in nonnull_counts.items() if v < .7] 
    data = data.drop(*to_drop)
    return data


# In[51]:

data = drop_null_columns(data)
print(data.count(),len(data.columns))


# ## Drop single distinct colummns

# In[53]:

count_cols = []
for col_name in data.columns:
    if data.select(col_name).distinct().count() == 1:
        count_cols.append(col_name)


# In[54]:

data = data.drop(*count_cols)
print(data.count(),len(data.columns))


# ## String to Float

# In[55]:

float_cols = ['masked columns for privacy']

float_f_cols = list(set(float_cols)-set(count_cols))

for col_name in float_f_cols:
    data = data.withColumn(col_name, col(col_name).cast('float'))


# In[56]:

numerical_cols= [item[0] for item in data.dtypes if item[1].startswith('float')]
numerical_data = data.select(numerical_cols)
print(numerical_data.count(),len(numerical_data.columns))


# ## Remove highly correlated columns

# In[59]:

from pyspark.mllib.stat import Statistics
import pandas as pd

## Correlation Matrix
features = numerical_data.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = numerical_cols, numerical_cols

# Select upper triangle of correlation matrix
upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]



# In[61]:

final_data = data.drop(*to_drop)
print(final_data.count(),len(final_data.columns))


# ## Drop columns with any null values

# In[63]:

from pyspark.sql.functions import isnan, when, count, col

def drop_columns(data):
    
    null_counts = final_data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c)  for c in final_data.columns]).collect()[0].asDict()
    to_drop = [k for k, v in null_counts.items() if v != 0] 
    data = data.drop(*to_drop)
    return data

final_data = drop_columns(final_data)



final_data.printSchema()


# ## Na Imputation


##### String and numerical columns

string_cols= ([item[0] for item in final_data.dtypes if (item[1].startswith('string')) & (item[0] not in ('base_plan_id','ant_drug_strength'))] )
numeric_cols  = [item[0] for item in final_data.dtypes if item[1].startswith('float')]



####Impute categorical columns
#import pyspark.sql.functions as F
#from pyspark.ml.feature import Imputer 

#def na_imputer(final_data,string_cols,numeric_cols):
    #for col_name_s in string_cols:
           # common = final_data.dropna().groupBy(col_name).agg(F.count("*")).orderBy('count(1)', ascending=False).first()[col_name]
           # final_data = final_data.withColumn(col_name, F.when(F.isnull(col_name), common).otherwise(df[col_name]))
        
    ####imputing Numerical columns
    
    #imputer = Imputer(inputCols=numeric_cols, outputCols=["{}_imputed".format(c) for c in numeric_cols])
    #imputer.fit(final_data).transform(final_data)
    #        
    #return(final_data)

#final_data = final_data.dropna()
#final_data = na_imputer(final_data,string_cols,numeric_cols)
#print(final_data.count(),len(final_data.columns))

#final_data.printSchema()


# ## Machine Learning Pipeline




from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer,VectorAssembler,OneHotEncoder,VectorIndexer
from pyspark.sql.functions import col
stages = []
for stringCols in string_cols:
    stringIndexer = StringIndexer(inputCol = stringCols, outputCol = stringCols + 'Index',handleInvalid='skip')
    encoder = OneHotEncoder(inputCol=stringIndexer.getOutputCol(), outputCol=stringCols + "stringEnc")
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol = 'base_plan_id', outputCol = 'label',handleInvalid='skip')
stages += [label_stringIdx]
assemblerInputs = [c + "stringEnc" for c in string_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]




import time
start_time = time.time()

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(final_data)
df = pipelineModel.transform(final_data)
end_time = time.time()
print("total time taken for Pipeline loop in seconds: ", end_time - start_time)

selectedCols = ['label', 'features'] + final_data.columns
df = df.select(selectedCols)
#df.printSchema()


# ## Random Forest Classification


from pyspark.ml.classification import RandomForestClassifier

### MinMax Scaling
from pyspark.ml.feature import MinMaxScaler
scaler = MinMaxScaler(inputCol='features',outputCol='scaledfeatures')

start_time = time.time()

scalermodel = scaler.fit(df)
scalerdata = scalermodel.transform(df)

end_time = time.time()
print("total time taken for Scaling loop in seconds: ", end_time - start_time)



train,test = scalerdata.randomSplit([0.8,0.2])
start_time = time.time()
rf=RandomForestClassifier(featuresCol="scaledfeatures",labelCol="label",predictionCol="prediction", probabilityCol="probability", rawPredictionCol="rawPrediction")
rfModel=rf.fit(train)
end_time = time.time()
print("total time taken to run rf in seconds: ", end_time - start_time)



predicted=rfModel.transform(test)



from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator=MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy")

accuracy=evaluator.evaluate(predicted)


print(accuracy)


from pyspark.mllib.evaluation import MulticlassMetrics
pred_label = predicted.select(['label','prediction'])

#confmat = pred_label.rdd.map(tuple)


confmat = pred_label.rdd.map(tuple)


metrics = MulticlassMetrics(confmat)
confusion_mat = metrics.confusionMatrix()



print(confusion_mat.toArray())




