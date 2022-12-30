# Databricks notebook source
# MAGIC %sh pip install geopandas tqdm

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from generate_matrix import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Fonctions utiles

# COMMAND ----------

def save_clean(df :pd.DataFrame, value :str, filename :str):
    df.columns = [title.lower() for title in df.columns]
    df = df[["longitude", "latitude", value.lower()]]
    df.to_csv(f"data/clean/{filename}.csv", index=False)

def save_map(map :np.ndarray, filename :str):
    np.save(f"output/{filename}.npy", map)

# COMMAND ----------

# MAGIC %md
# MAGIC # Objets importants

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Carte de référence (xMap)
# MAGIC 
# MAGIC <i> La carte de référence sous tends le modele géographique, il sert de "cadre" pour délimiter les lieux étudiés et définit la latitude & longitude pour chacun des éléments de sa matrice. C'est une matrice de dimension (52, 118, 2) (hauteur, longueur, [latitude, longitude])  </i>

# COMMAND ----------

xmap = np.load("output/us_map.npy")

font1 = {'size'   : 22}
font2 = {'size' : 16}

fig, (lat, lon) = plt.subplots(1, 2)
fig.set_size_inches(25, 5)

fig.suptitle("xMap", **font1)
lat.set_title("Latitude", **font2)
lat.imshow(xmap[:,:,0], cmap="Greys");
lon.set_title("Longitude", **font2)
lon.imshow(xmap[:,:,1], cmap="Greys");

# COMMAND ----------

xmap.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jurisdictions

# COMMAND ----------

us_landmass = np.load("output/us_landmass.npy")
can_landmass = np.load("output/can_landmass.npy")
mex_landmass = np.load("output/mex_landmass.npy")

fig, (us, can, mex) = plt.subplots(1, 3)

fig.set_size_inches(35, 5)
fig.suptitle("Jurisdictions", **font1)
us.set_title("Américaine", **font2)
us.imshow(us_landmass, cmap="terrain_r");
can.set_title("Canadienne", **font2)
can.imshow(can_landmass, cmap="terrain_r");
mex.set_title("Mexicaine", **font2)
mex.imshow(mex_landmass, cmap="terrain_r");

# COMMAND ----------

# MAGIC %md
# MAGIC # Données

# COMMAND ----------

# MAGIC %md
# MAGIC ## Données géographiques

# COMMAND ----------

# MAGIC %py 
# MAGIC 
# MAGIC fig, axs = plt.subplots(3, 4)
# MAGIC fig.set_size_inches(40, 12)
# MAGIC fig.suptitle("Données géographiques", **font1)
# MAGIC 
# MAGIC # landmass 
# MAGIC landmass = us_landmass + can_landmass + mex_landmass
# MAGIC axs[0,0].set_title("Territoire", **font2)
# MAGIC axs[0,0].imshow(landmass, cmap="terrain_r")
# MAGIC 
# MAGIC # airports
# MAGIC airports = np.load("output/airports.npy")
# MAGIC axs[0,1].set_title("Aéroports internationaux", **font2)
# MAGIC axs[0,1].imshow(landmass - airports, cmap="terrain_r")
# MAGIC 
# MAGIC # cng stations
# MAGIC cng = np.load("output/cng_stations.npy")
# MAGIC axs[0,2].set_title("Stations de gaz naturel compressé", **font2)
# MAGIC axs[0,2].imshow(landmass - cng, cmap="terrain_r")
# MAGIC 
# MAGIC # docks
# MAGIC docks = np.load("output/docks.npy")
# MAGIC axs[0,3].set_title("Quais", **font2)
# MAGIC axs[0,3].imshow(landmass - docks, cmap="terrain_r")
# MAGIC 
# MAGIC # e85 stations
# MAGIC e85 = np.load("output/e85_stations.npy")
# MAGIC axs[1,0].set_title("Stations superéthanol-E85", **font2)
# MAGIC axs[1,0].imshow(landmass - e85, cmap="terrain_r")
# MAGIC 
# MAGIC # electric stations
# MAGIC elec = np.load("output/elec_stations.npy")
# MAGIC axs[1,1].set_title("Bornes de recharge", **font2)
# MAGIC axs[1,1].imshow(landmass*30 - elec, cmap="terrain_r")
# MAGIC 
# MAGIC # ferry terminals
# MAGIC ferry = np.load("output/ferry_terminals.npy")
# MAGIC axs[1,2].set_title("Ports pour bateau de courte distance", **font2)
# MAGIC axs[1,2].imshow(landmass*5 - ferry, cmap="terrain_r")
# MAGIC 
# MAGIC # lng terminals
# MAGIC lng = np.load("output/lng_terminals.npy")
# MAGIC axs[1,3].set_title("Terminaux de gaz naturel liquéfié", **font2)
# MAGIC axs[1,3].imshow(landmass*2 - lng, cmap="terrain_r")
# MAGIC 
# MAGIC # lpg stations
# MAGIC lpg = np.load("output/lpg_stations.npy")
# MAGIC axs[2,0].set_title("Stations de gaz de pétrole liquéfié", **font2)
# MAGIC axs[2,0].imshow(landmass - lpg, cmap="terrain_r")
# MAGIC 
# MAGIC #railroad nodes
# MAGIC rlrd = np.load("output/rlrd_nodes.npy")
# MAGIC axs[2,1].set_title("Stations et arrêts de train", **font2)
# MAGIC axs[2,1].imshow(landmass - rlrd, cmap="terrain_r")
# MAGIC 
# MAGIC #transit stops
# MAGIC stops = np.load("output/transit_stops.npy")
# MAGIC axs[2,2].set_title("Arrêts transports en commun", **font2)
# MAGIC axs[2,2].imshow(landmass*30 - stops, cmap="terrain_r")
# MAGIC 
# MAGIC #truck stop parkings
# MAGIC truck = np.load("output/truck_stops.npy")
# MAGIC axs[2,3].set_title("Arrêts de camions", **font2)
# MAGIC axs[2,3].imshow(landmass*30 - truck, cmap="terrain_r")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Données locales
# MAGIC 
# MAGIC <i> Aller voir fichier intitulé classification_lieux pour voir les opérations ayant étés produites pour localiser les lieux étudiés </i>

# COMMAND ----------

mkts = spark.sql('select * from hive_metastore.gs.market_sectors__historical_market').toPandas()

# COMMAND ----------

mkts_ref = pd.read_csv("data/clean/markets_reference.csv")
mkts_ref["val"] = 1
mkts_ref

# COMMAND ----------

markets = populate(xmap, mkts_ref.longitude.to_list(), mkts_ref.latitude.to_list(), mkts_ref.val.to_list())[::-1]
plt.figure(figsize=(20, 6))
plt.title("Marchés étudiés", **font1)
plt.imshow(landmass - markets, cmap="terrain_r");

# COMMAND ----------

mkts_indus = mkts[mkts["sector"]=="Industrial"]
mkts_indus = mkts_indus[["market_publish", "age_median", "airport_volume", "asset_value_momentum", "desirability_quintile", "fiscal_health_tax_quintile", "interstate_distance", "interstate_miles", "mrevpaf_growth_yoy_credit", "occupancy", "population_500mi"]]

# COMMAND ----------

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

# COMMAND ----------

mkts_indus.reset_index(drop=True, inplace=True)
mkts_indus

# COMMAND ----------


