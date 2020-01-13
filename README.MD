**Motivation:**

I have been living in Chicago for 5 years. Crime in Chicago is also a very interesting topic for exploration. Though I am a quiet indoor person who may not be able to feel the crime situation, the City of Chicago provides a very neat and detailed data on [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2019/w98m-zvie). Data from 2016 to the present is used for exploration.

The goal of this project is data visualization using popular tools recent days and time series analysis for prediction on the future. 

**Data exploration**

For a quick exploration, I want to know the Top 10 crimes and dangerous districts. So in the future, people can pay more caution to certain crimes or certain regions. Among those results, the rank of robbery draws my attention. My expectation is that robbery should rank after the theft as the top 2 crimes. But it seems criminals here are more interested in doing damage to people or things. So carrying cash for self-protection may not be very effective overall. The dangerous regions (by comparing to the district zone # [here](https://home.chicagopolice.org/office-of-community-policing/community-map/)) are Chicago downtown and southwestern area. 

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Top10_category.png)

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Top10_district.png)

The crime data is also very time-dependent. In the winter, the crime is less active probably due to the harsh weather. There are certain days in a week such as Monday and Friday that crime total number is higher. In one day, the trend is more obvious. A first peak occurs at noon and then a second around  6 pm. If you are a tourist, you may need to pay attention to your belongings during those periods. 

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_month.png)
![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_day.png)
![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Each_hour.png)

**Visualization**

The results above reveal the general trend for the crime. But to provide a better view, I created a heatmap on robbery amount in a day in 2019.  We can clearly observe the difference between the heatmap at 6 am and 6 pm. 

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/heatmap_by_time_6am.png) ![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/heatmap_by_time_6pm.png)

**Modelling**

I applied the k-means algorithm to the crime (robbery) location.

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/kmeans.png)

To predict the trend of crime number for this year, I used an ARIMA model for time series analysis. Generally, crime is decreasing which is good news. Lower crimes brings This means Chicago is becoming a better environment for investment. 

![image](https://github.com/RuiyunHuang/Chicago_Crime_Analysis/blob/master/figures/Predictions.png)

**Conclusion**

A quick analysis of the Chicago crime record provides some useful information on general living in the city. While it is not a comprehensive result. Improvement can be made by analyzing other features such as the detailed location and resolution result on the original file. It will give more insight to industry as well as government policy adjustment.
