# Databricks notebook source
# MAGIC %sh pip install tensorflow pandas_datareader

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt

import os


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Compiling the data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geo

# COMMAND ----------

maps = os.listdir("ml/data/map")
maps.remove("us_map.npy")
maps_dict = {map.replace(".npy",""): np.load(f"ml/data/map/{map}") for map in maps}
xmap = np.load("ml/data/map/us_map.npy")[::-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Local

# COMMAND ----------

mkts = spark.sql('select * from hive_metastore.gs.market_sectors__historical_market').toPandas()
mkts_ref = pd.read_csv("data/clean/markets_reference.csv")

# COMMAND ----------

mkts_indus = mkts[(mkts["sector"]=="Industrial")]
mkts_indus = mkts_indus[["market_publish", "date_bom", "age_median", "airport_volume", "asset_value_momentum", "desirability_quintile", "fiscal_health_tax_quintile", "interstate_distance", "interstate_miles", "mrevpaf_growth_yoy_credit", "occupancy", "population_500mi"]]
mkts_indus.columns = ["market_publish", "date", "age_median", "airport_volume", "asset_value_momentum", "desirability_quintile", "fiscal_health_tax_quintile", "interstate_distance", "interstate_miles", "mrevpaf_growth_yoy_credit", "occupancy", "population_500mi"]
mkts_indus.date = pd.to_datetime(mkts_indus.date)
mkts_indus = mkts_indus[(mkts_indus["date"].dt.month == 1)]

start = mkts_indus.date.to_list()[0] - dt.timedelta(days = 31)
end = mkts_indus.date.to_list()[-1]

mkts_indus.date = mkts_indus.date.dt.year


desirability_quintile_dict = {
    "Very Desirable" : 5,
    "Desirable" : 4,
    "Somewhat Desirable" : 3,
    "Less Desirable" : 2,
    "Much Less Desirable" : 1
}
mkts_indus.desirability_quintile = mkts_indus.desirability_quintile.apply(lambda quint : desirability_quintile_dict[quint])

fiscal_health_tax_quintile_dict = {
    "Healthy" : 3,
    "Stable" : 2,
    "Concerning" : 1
}
mkts_indus.fiscal_health_tax_quintile = mkts_indus.fiscal_health_tax_quintile.apply(lambda quint : fiscal_health_tax_quintile_dict[quint])

# COMMAND ----------

def get_lon(location :str):
    return mkts_ref.loc[mkts_ref["market_publish"] == location]["longitude"].item()

def get_lat(location :str):
    return mkts_ref.loc[mkts_ref["market_publish"] == location]["latitude"].item()

mkts_indus["latitude"] = mkts_indus["market_publish"].apply(get_lat)
mkts_indus["longitude"] = mkts_indus["market_publish"].apply(get_lon)

# COMMAND ----------

#Pour gérer les valeurs manquantes, on pourrait remplacer par la moyenne/médianne. Toutefois, ce serait mieux d'y aller par proximité (remplacer par la valeur de lieux le plus proche). Par souci de temps j'y vais par la médiane
mkts_indus.fillna(mkts_indus.median(), inplace=True)
mkts_indus = mkts_indus.groupby(["market_publish", "date"]).first()
mkts_indus

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Macro

# COMMAND ----------

var_macro = ['dcoilwtico','dexmxus','dexuseu', 'dexcaus', 'netexp']



var_macro_dict = {}
var_macro_df = pd.DataFrame()

for i in var_macro:
    # téléchargement sur la fed américaine à l'aide de pandas datareader
    var_macro_dict[i] = web.DataReader(i.upper(), 'fred', start, end)
    if var_macro_df.empty:
        var_macro_df = var_macro_dict[i]
    else:
        var_macro_df = var_macro_df.join(var_macro_dict[i])

# COMMAND ----------

var_macro_df = var_macro_df.ffill().dropna().resample("Y").first()
var_macro_df.index = var_macro_df.index.year
var_macro_df.index.names = ['date']


# COMMAND ----------

# MAGIC %md
# MAGIC ### NCF Growth

# COMMAND ----------

ncf = spark.sql('select * from hive_metastore.gs.forecasts__historical_baseline').toPandas()

# COMMAND ----------

ncf.date = pd.to_datetime(ncf.date)
ncf = ncf[(ncf.date_fc_release == "2022-03-31") & (ncf.sector_publish == "Industrial") & (ncf.date >= start + dt.timedelta(days=30*12)) & (ncf.date <= end) & (ncf.market_publish != "Top 50")]
ncf.date = ncf.date.dt.year

# COMMAND ----------

ncf = ncf[["market_publish", "date", "ncf_growth"]].groupby(["market_publish", "date"]).last()

# COMMAND ----------

# MAGIC %md
# MAGIC # Train and test datasets

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Geo data

# COMMAND ----------

def get_zone(longitude, latitude, map, size = 10):
    x, y = np.unravel_index(np.argmin(np.sqrt((xmap[:, :, 0] - latitude) ** 2 + (xmap[:, :, 1] - longitude) ** 2)), xmap.shape[:2])
    zone = maps_dict[map][max([x-size, 0]):min([x+size, xmap.shape[0]]), max([y-size,0]):min([y+size,xmap.shape[1]])]
    if (x-size < 0):
        top = -(x-size)
        zone = np.append(np.zeros((top, zone.shape[1])), zone, axis=0)
    elif (x+size > xmap.shape[0]):
        bottom = x + size - xmap.shape[0]
        zone = np.append(zone, np.zeros((bottom, zone.shape[1])), axis=0)
    
    if (y-size < 0):
        left = -(y-size)
        zone = np.append(np.zeros((zone.shape[0],left)), zone, axis = 1)
    elif (y+size > xmap.shape[1]):
        right = y + size - xmap.shape[1]
        zone = np.append(zone, np.zeros((zone.shape[0],right)), axis = 1)
    return zone.astype('float64')
    

# COMMAND ----------

for map in maps_dict.keys():
    ncf[map] = None
    
for row in ncf.index:
    city = row[0]
    lon = get_lon(city)
    lat = get_lat(city)
    for map in maps_dict.keys():
        ncf[map][row] = get_zone(lon, lat, map, 15)
        

# COMMAND ----------

LON = -87.6244212
LAT = 41.8755616
MAP = "ferry_terminals"
SIZE = 15
plt.title(f"{MAP} at lat = {round(LAT,2)} & lon = {round(LON,2)} with zoom of {SIZE}")
plt.imshow(get_zone(LON, LAT, MAP, SIZE)+ get_zone(LON, LAT, "us_landmass", SIZE)*10);

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Local data

# COMMAND ----------

ncf_ = ncf.join(mkts_indus)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Macro data

# COMMAND ----------

ncf__ = ncf_.join(var_macro_df)

# COMMAND ----------

ncf__
