# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### In this notebook, you can learn how to use Spark SQL for big data analysis on SF crime data. (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry). 
# MAGIC The first part of Homework is OLAP for scrime data analysis (80 credits).  
# MAGIC The second part is unsupervised learning for spatial data analysis (20 credits).   
# MAGIC The option part is the time series data analysis (50 credits).  
# MAGIC **Note**: you can download the small data (one month e.g. 2018-10) for debug, then download the data from 2013 to 2018 for testing and analysising. 
# MAGIC 
# MAGIC ### How to submit the report for grading ? 
# MAGIC Publish your notebook and send your notebook link to mike@laioffer.com. 
# MAGIC Your report have to contain your data analysis insights.  
# MAGIC 
# MAGIC ### Deadline 
# MAGIC Two weeks from the homework release date
# MAGIC 
# MAGIC ### cluster 创建
# MAGIC 创建cluster 的时候选择python 3  
# MAGIC 创建cluster 的时候选择python 3  
# MAGIC 创建cluster 的时候选择python 3  
# MAGIC 
# MAGIC ### 画图
# MAGIC 使用Databricks 自带的画图就好了，不要求使用其他的工具
# MAGIC 
# MAGIC ### Time series 
# MAGIC 不讲，个人随意

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

# 从SF gov 官网读取下载数据
# 不要反复执行，大家执行一次就好了啊
# 第二次记得comment 掉
#import urllib.request
#urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/sf_03_18.csv")
#dbutils.fs.mv("file:/tmp/sf_03_18.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
#display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))
## 或者自己下载
# https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD


# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# DBTITLE 1,Data preprocessing
# read data from the data storage
# please upload your data into databricks community at first. 
crime_data_lines = sc.textFile(data_path)
#prepare data 
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])
#get header
header = df_crimes.first()
print(header)

#remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)

#get the first line of data
#display(crimes.take(3))

#get the total number of data 
print(crimes.count())


# COMMAND ----------

# MAGIC %md
# MAGIC ### Solove  big data issues via Spark
# MAGIC approach 1: use RDD (not recommend)  
# MAGIC approach 2: use Dataframe, register the RDD to a dataframe (recommend for DE)  
# MAGIC approach 3: use SQL (recomend for data analysis or DS， 基础比较差的同学)  
# MAGIC ***note***: you only need to choose one of approaches as introduced above

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
# remove non-criminal type
df_opt1=df_opt1.where('category != "NON-CRIMINAL" and category != "MISSING PERSON"')
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.
# MAGIC 
# MAGIC Below are some example codes to demonstrate the way to use Spark RDD, DF, and SQL to work with big data. You can follow this example to finish other questions. 

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False).limit(10)
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC limit 10")
display(crimeCategory)

# COMMAND ----------

# DBTITLE 1,Visualize your results
# important hints: 
## first step: spark df or sql to compute the statisitc result 
## second step: export your result to a pandas dataframe. 

crimes_pd_df = crimeCategory.toPandas()

# Spark does not support this function, please refer https://matplotlib.org/ for visuliation. You need to use display to show the figure in the databricks community. 

#display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results

# COMMAND ----------

#check pddistrict is null. I ignored here since there is only one case, but it need to replace true value with X,Y.
crimed_dt=spark.sql("select * from sf_crime where pddistrict is null")
display(crimed_dt)

# COMMAND ----------

# use spark sql
crimed_dt=spark.sql("select pddistrict as district, count(*) as number from sf_crime where pddistrict is not null group by pddistrict order by 1")
df_dt=crimed_dt.toPandas()
display(df_dt)

# COMMAND ----------

# use spark dataframe(still sql command in 'where')
crimed_dt_c=df_opt1.where('pddistrict is not null').groupBy('pddistrict').count().orderBy('count')
display(crimed_dt_c)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3 question (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC hints: SF downtown is defiend  via the range of spatial location. For example, you can use a rectangle to define the SF downtown, or you can define a cicle with center as well. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/

# COMMAND ----------

#define a circle for selecting SFDT for spark sql and dataframe
from pyspark.sql.types import IntegerType
from pyspark.sql import functions as F
def is_sfdt(x, y):
  x=float(x)
  y=float(y)
  if (y-37.793984)**2+(x+122.401351)**2 < 0.000047519:
    return 1
  else:
    return 0
spark.udf.register("is_in_sfdt", is_sfdt, IntegerType())
is_in_sfdt_udf = udf(is_sfdt, IntegerType())

# COMMAND ----------

# use spark sql
crime_sfdt=spark.sql("select TO_DATE(CAST(UNIX_TIMESTAMP(date, 'MM/dd/yyyy') AS TIMESTAMP)) as date, count(*) as number_of_crime from sf_crime where date is not null and dayofweek='Sunday' and is_in_sfdt(x, y) > 0 group by 1 order by 1")
df_sfdt=crime_sfdt.toPandas()
display(df_sfdt)

# COMMAND ----------

#use dataframe without sql command
crime_sfdt_c=df_opt1.where((is_in_sfdt_udf('x', 'y')>0) & (df_opt1.DayOfWeek == 'Sunday')).withColumn('date', F.to_date(df_opt1.Date, 'MM/dd/yyyy')).groupby('date').count().orderBy('date')
display(crime_sfdt_c)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

years={'2015', '2016', '2017', '2018'}
crime_by_month=df_opt1.withColumn('year', F.regexp_replace(df_opt1.Date,'(\d+)/(\d+)/(\d+)', '$3')).where(F.col('year').isin(years)).withColumn('month', F.regexp_replace(df_opt1.Date,'(\d+)/(\d+)/(\d+)', '$1')).groupBy('year', 'month').count().orderBy('year', 'month')
display(crime_by_month)
# according to the result, the crime numbers in each month from 2015-2017 and crime numbers from month to month are similar. While in 2018, there is an obvious decreases in first 5 months with last month mostly due to incomplete recording. 
# for business, due to the decrease of crime number, investors are expected to allocate less money in security while invest more in attracting customers or product development. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

days={'12/15/2017', '12/15/2016', '12/15/2015'}
crime_in_a_day=df_opt1.where(df_opt1.Date.isin(days)).withColumn('hour', F.regexp_replace(df_opt1.Time, '(\d+):(\d+)', '$1')).groupBy('Date','hour').count().orderBy('Date','hour')
display(crime_in_a_day)
#The graph shows the lowest crime count occurs from 1 - 7 am very likely due to low human activities. This is not very relavant information for travel since most places are closed unless you like bars or nightclubs(however you would expect more crimes in those places). For a normal tourist, the time to avoid lots of travel or need higher alert are around noon and 6-7 pm since those shows highest crime numbers.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

dangerous_district={'SOUTHERN','MISSION','NORTHERN'}
crime_in_top_three_region=df_opt1.where(df_opt1.PdDistrict.isin(dangerous_district)).withColumn('hour', F.regexp_replace(df_opt1.Time, '(\d+):(\d+)', '$1')).groupBy('category', 'hour').count().orderBy('category', 'hour', 'count')
display(crime_in_top_three_region)
# According to result Q2, I decided to use the highest three fraction in the pie figures for top-3 danger district. A better decision could be made by using the crime number/density of populations in urban areas to alleviate the influence of district areas. 
# According to the result, theft is a major portion of total crimes. This type of crime is usually expected to happen in area with higher population density like public transportations or shopping center or malls. So it is suggested to dispatch more police into those areas. Still, there is a time dependence on the crime, so less patrol is needed overnight but more is needed around noon and evening everyday.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

crime_solution_percentage=df_opt1.withColumn('resolution', F.when(df_opt1.Resolution =='NONE', 'NONE').when(df_opt1.Resolution == 'ARREST, BOOKED', 'ARREST, BOOKED').otherwise('OTHERS')).groupBy('category', 'resolution').count().orderBy('category', 'count')
display(crime_solution_percentage)
#Major type of resolutions are none and arrested for most cases, thus I grouped all the other types as 'others' for clearer view. For detail analysis, we can concentrated on one type of crime for further analysis. In different types of crimes, the major resolution is none. This may be the results of different reasons, like the judge of police or the difficulty of solving a case. For general policy adjustment, it is till necessary to increase the efficiency of the policy since there are too many none type resolution. This maybe achieved by relating the salary to finalized cases numbers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Use four sentences to summary your work. Like what you have done, how to do it, what the techinical steps, what is your business impact. 
# MAGIC More details are appreciated. You can think about this a report for your manager. Then, you need to use this experience to prove that you have strong background on big  data analysis.  
# MAGIC Point 1:  what is your story ? and why you do this work ?   
# MAGIC Point 2:  how can you do it ?  keywords: Spark, Spark SQL, Dataframe, Data clean, Data visulization, Data size, clustering, OLAP,   
# MAGIC Point 3:  what do you learn from the data ?  keywords: crime, trend, advising, conclusion, runtime 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Optional part: Time series analysis
# MAGIC This part is not based on Spark, and only based on Pandas Time Series package.   
# MAGIC Note: I am not familiar with time series model, please refer the ARIMA model introduced by other teacher.   
# MAGIC process:  
# MAGIC 1.visualize time series  
# MAGIC 2.plot ACF and find optimal parameter  
# MAGIC 3.Train ARIMA  
# MAGIC 4.Prediction 
# MAGIC 
# MAGIC Refer:   
# MAGIC https://zhuanlan.zhihu.com/p/35282988  
# MAGIC https://zhuanlan.zhihu.com/p/35128342  
# MAGIC https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_0.html  
# MAGIC https://www.howtoing.com/a-guide-to-time-series-forecasting-with-arima-in-python-3  
# MAGIC https://www.joinquant.com/post/9576?tag=algorithm  
# MAGIC https://blog.csdn.net/u012052268/article/details/79452244  
