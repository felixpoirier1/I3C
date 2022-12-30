# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# COMMAND ----------

forecasts = spark.sql('select * from hive_metastore.gs.forecasts__historical_baseline').toPandas()

# COMMAND ----------

miami = forecasts[(forecasts.date_fc_release == "2021-03-31") & (forecasts.sector_publish == "Industrial") & (forecasts.market_publish == "Miami")]
miami.head()

# COMMAND ----------

plt.figure(figsize=(50, 20))
plt.title("mrevpaf_growth")
miami = forecasts[(forecasts.fc_publish == "2021 Q1") & (forecasts.sector_publish == "Industrial") & (forecasts.market_publish == "Miami")]
plt.plot(miami.date, miami.supply_growth, label = f"{miami.fc_publish.to_list()[0]}")
miami = forecasts[(forecasts.date_fc_release == "2022-03-31") & (forecasts.sector_publish == "Industrial") & (forecasts.market_publish == "Miami")]
plt.plot(miami.date, miami.supply_growth, label = f"{miami.fc_publish.to_list()[0]}");
plt.legend()
plt.grid(True)

# COMMAND ----------

miami.date_fc_release.to_list()

# COMMAND ----------

forecasts.describe()

# COMMAND ----------

past_data = forecasts[(forecasts.date_fc_release == "2022-03-31") & (forecasts.sector_publish == "Industrial")]

# COMMAND ----------

past_data

# COMMAND ----------

cities_df = pd.read_csv("data/clean/american_cities.csv")
cities = cities_df.city.tolist()[:10]+ ["Top 50"]
past_data.date = pd.to_datetime(past_data.date)
past_data = past_data[past_data.date.dt.month == 12]

plt.figure(figsize=(50, 20), facecolor="white")
plt.title("Net Cash Flow Growth of 10 most populous american cities", **{"size": 34})

for city in cities:
    data = past_data[(forecasts.market_publish == city)]
    if city == "Top 50":
        plt.plot(data.date, data.ncf_growth.cumsum(), linewidth=7.0, label = city);
    else:
        plt.plot(data.date, data.ncf_growth.cumsum(), linewidth=2.0, label = city);

plt.text(pd.Timestamp('2024-03-31 00:00:00'), 1.25, "Forecast", **{"size": 20})
plt.text(pd.Timestamp('2012-03-31 00:00:00'), 1.25, "Historical", **{"size": 20})

plt.xlabel("Date", **{"size": 20})
plt.ylabel("Cumulative growth in percentage", **{"size": 20})


forecast = data.date.loc[39851:].to_list()
plt.axvspan(forecast[0], forecast[-1], facecolor='r', alpha=0.1)
plt.legend(fontsize=15)
plt.savefig("img/NCF_Top10.png")

# COMMAND ----------

past_data = past_data[past_data.date.dt.month == 12]
past_data

# COMMAND ----------


