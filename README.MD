**Motivation:**

I have been living in Chicago for 5 years. Crime in Chicago is always a very interesting topic for exploration. The City of Chicago provides a very neat and detailed data on [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2019/w98m-zvie). Data from 2016 to the present is used for exploration.

The goal of this project is to provide insights for personal safety suggestions via space and time series analysis of crimes. 

**Data exploration**

It is useful to know the Top 10 crimes and dangerous districts. So people can pay more caution to certain crimes or certain regions. Among those results, the rank of robbery is surprising. Generally speaking, I think robbery should rank only after the theft since I expect criminals would have a practical motivation to commit a crime. My guess on the current observation is that most crimes involving personal harm may be failed robbery, such as battery and assault. Based on that, I think carrying cash for self-protection may still be effective to avoid body violence. 

The dangerous regions (by comparing to the district zone # [here](https://home.chicagopolice.org/office-of-community-policing/community-map/)) are Chicago downtown and southwestern area. This result is common sense for people living here for a long time. 

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Top10_category.png)

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Top10_district.png)

The crime data is also very time-dependent. In the winter, the crime is less active probably due to the harsh weather. There are certain days in a week such as Monday and Friday that crime total number is higher. In one day, the trend is more obvious. A first peak occurs at noon and then a second peak is observed at around  6 pm. If you are a tourist, you may need to pay attention to your belongings during those periods. 

**By month**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_month.png)

**By day of week**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_day.png)

**By hour in a day**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_hour.png)

**Visualization**

The results above reveal the general spatial and time distribution of crime. To provide a better result, a heatmap is used.  The heatmap shows the robbery distribution within one day in 2019. We can clearly observe the difference between 6 am and 6 pm. 

**6am**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/heatmap_by_time_6am.png)

**6pm**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/heatmap_by_time_6pm.png)

**Modelling**

 The k-means algorithm is used for classifying the crime (robbery) location. The result shows the region with the highest number of robbery is the downtown and southwestern area of Chicago. It is consistent with the previous analysis.

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/kmeans.png)

To predict the crime number, I used a SARIMA model for time series analysis first. The data is split into training data (2016-2018) and testing data(2019). This model does not give a good prediction due to the dual seasonality in the data. So a TBATS model is then used to solve the problem. 

**SARIMAX**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/SARIMAX.png)

**TBATS**

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/TBATS.png)


**Conclusion**

A quick analysis of the Chicago crime record provides some useful information on general living in the city. While it is not a comprehensive result. Improvement can be made by analyzing other features such as the detailed location and resolution result. It will give more insight to industry as well as government policy adjustment.
