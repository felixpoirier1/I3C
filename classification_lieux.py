# Databricks notebook source
# MAGIC %sh pip install geopandas tqdm geopy

# COMMAND ----------

from generate_matrix import *
from coordinates import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC # Lieux (Markets)

# COMMAND ----------

mkts = spark.sql('select * from hive_metastore.gs.market_sectors__historical_market').toPandas()

# COMMAND ----------

mkt_ref = mkts[["market_publish", "longitude", "latitude"]].groupby("market_publish").first()
print(f'{mkt_ref.shape[0] - mkt_ref.dropna().shape[0]} missing locations')
mkt_ref

# COMMAND ----------

# MAGIC %md
# MAGIC Il manque 47 lieux à identifier en latitude et longitude, voici la séquence avec laquelle nous allons remplir ces valeurs manquantes
# MAGIC 
# MAGIC 1. Se référer au fichier ~/data/clean/american_cities.csv, dans lequel ce trouve environ 1000 villes américaines et essayer de trouver un "match"
# MAGIC 2. Si aucun match utiliser la fonction fetch_coordinates() contenue dans le fichier coordinates.py
# MAGIC 3. Si ceci ne fonctionne pas, simplement drop la valeur

# COMMAND ----------

empty_mkt_df = mkt_ref.isna()
empty_mkt_df.drop(empty_mkt_df[~(empty_mkt_df.longitude) & ~(empty_mkt_df.latitude)].index, inplace=True)
empty_mkt = list(empty_mkt_df.index)
empty_mkt

# COMMAND ----------

cities = pd.read_csv("data/clean/american_cities.csv")
cities = cities.groupby("city").first()
print(cities.index)
tbr = []
for mkt in empty_mkt:
    if mkt in cities.index:
        mkt_ref.at[mkt, "latitude"] = cities.loc[mkt]["lat"]
        mkt_ref.at[mkt, "longitude"] = cities.loc[mkt]["lng"]
        tbr += [mkt]
    else:
        try:
            latlon = fetch_coordinates(mkt, "US")
            mkt_ref.at[mkt, "latitude"] = latlon[0]
            mkt_ref.at[mkt, "longitude"] = latlon[1]
        except:
            mkt_ref.drop(mkt)
            

            
        
for i in tbr:
    empty_mkt.remove(i)

# COMMAND ----------

mkt_ref = mkt_ref.dropna().drop("Top 50")
mkt_ref

# COMMAND ----------

mkt_ref.to_csv("data/clean/markets_reference.csv")
