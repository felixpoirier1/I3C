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
import seaborn as sns

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

mkts.groupby("sector").count()

# COMMAND ----------

mkts_indus = mkts[(mkts["sector"]=="Industrial")]
mkts_indus = mkts_indus[["market_publish", "date_bom", "age_median", "airport_volume", "asset_value_momentum", "desirability_quintile", "fiscal_health_tax_quintile", "interstate_distance", "interstate_miles", "mrevpaf_growth_yoy_credit", "occupancy", "population_500mi"]]
mkts_indus.columns = ["market_publish", "date", "age_median", "airport_volume", "asset_value_momentum", "desirability_quintile", "fiscal_health_tax_quintile", "interstate_distance", "interstate_miles", "mrevpaf_growth_yoy_credit", "occupancy", "population_500mi"]
print(mkts_indus)
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
    var_macro_dict[i] = web.DataReader(i.upper(), 'fred', start, end + dt.timedelta(days=365))
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

ncf

# COMMAND ----------

ncf.date = pd.to_datetime(ncf.date)
# & ((ncf.date <= end)|((ncf.date.dt.year == end.year) & (ncf.date.dt.month == 12)))

ncf = ncf[(ncf.date_fc_release == "2022-03-31") & ((ncf.date <= end) | ((ncf.date.dt.year == end.year) & (ncf.date.dt.month == 12))) & (ncf.sector_publish == "Industrial")  & (ncf.market_publish != "Top 50")]

ncf.date = ncf.date.dt.year

# COMMAND ----------

end

# COMMAND ----------

ncf = ncf[["market_publish", "date", "ncf_growth"]].groupby(["market_publish", "date"]).last()

# COMMAND ----------

ncf["ncf_growth_1"] = ncf.ncf_growth.shift(1)
ncf["ncf_growth_2"] = ncf.ncf_growth.shift(2)
ncf["ncf_growth_3"] = ncf.ncf_growth.shift(3)

# COMMAND ----------

ncf = ncf.reset_index()
ncf = ncf.loc[(ncf.date >= 2019)].groupby(["market_publish", "date"]).first()
ncf

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
    return zone.astype(np.float32)
    

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

LON = -74.040900
LAT = 40.762699
MAP = "lpg_stations"
SIZE = 15

plt.figure(figsize=(10, 10))

plt.title(f"{MAP} at lat = {round(LAT,2)} \n& lon = {round(LON,2)} with zoom of {SIZE}", **{"size": 20})
plt.imshow(get_zone(LON, LAT, MAP, SIZE)+ get_zone(LON, LAT, "us_landmass", SIZE));
plt.axis('off')

plt.savefig("img/zoom_15_lpg.png")

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

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Splitting dataset

# COMMAND ----------

x_df = ncf__.copy().drop(["latitude", "longitude"], axis=1)
new_order = x_df.columns[4:].append(x_df.columns[:4])

x_pred = x_df[new_order].drop([2019, 2020, 2021], level=1, axis=0)
y_pred = x_pred.pop("ncf_growth")
x_df = x_df[new_order].drop(2022, level=1, axis=0)
y_df = x_df.pop("ncf_growth")

x_arr = np.array(x_df)
y_arr = np.array(y_df)

x_pred_arr = np.array(x_pred)
y_pred_arr = np.array(y_pred)

labelname = np.array(y_pred.index.levels[0])

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.33, random_state=1)

X_train_val = X_train[:,14:].astype(np.float32)
X_train_val = tf.keras.utils.normalize(X_train_val)
X_train_mat = X_train[:, :14]
X_train_mat = np.stack(X_train_mat.tolist())
X_train_mat = tf.keras.utils.normalize(X_train_mat)

X_test_val = X_test[:,14:].astype(np.float32)
X_test_val = tf.keras.utils.normalize(X_test_val)
X_test_mat = X_test[:, :14]
X_test_mat = np.stack(X_test_mat.tolist())
X_test_mat = tf.keras.utils.normalize(X_test_mat)

X_pred_val = x_pred_arr[:,14:].astype(np.float32)
X_pred_val = tf.keras.utils.normalize(X_pred_val)
X_pred_mat = x_pred_arr[:, :14]
X_pred_mat = np.stack(X_pred_mat.tolist())
X_pred_mat = tf.keras.utils.normalize(X_pred_mat)

# COMMAND ----------

X_train_mat

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Neural Network

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Creating model structure

# COMMAND ----------

np.random.seed(42)
tf.random.set_seed(42)

# COMMAND ----------

model = None

# COMMAND ----------

input_map = keras.layers.Input(shape=X_train_mat.shape[1:], name="geo_input")
input_map_flat = keras.layers.Flatten()(input_map)
hidden_map_1 = keras.layers.Dense(900, activation="relu")(input_map_flat)
hidden_map_2 = keras.layers.Dense(1000, activation="relu")(hidden_map_1)
hidden_map_3 = keras.layers.Dense(500, activation="relu")(hidden_map_2)

input_B = keras.layers.Input(shape=X_train_val.shape[1:], name="local_macro_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([hidden_map_3, hidden2])
hidden3 = keras.layers.Dense(900, activation="relu")(concat)
hidden4 = keras.layers.Dense(30, activation="relu")(hidden3)
output = keras.layers.Dense(1, name="output")(hidden4)
model = keras.models.Model(inputs=[input_map, input_B], outputs=[output])

model.compile(loss="mean_squared_error", optimizer="sgd")

model.summary()

# COMMAND ----------

history = model.fit([X_train_mat, X_train_val], [y_train, y_train], epochs=100)

# COMMAND ----------

plt.plot(history.history['loss'])

# COMMAND ----------

bm = model.predict((X_train_mat, X_train_val))


# COMMAND ----------

real = np.transpose([y_train])

x_base = [i/100 for i in list(range(-10, 20))]

plt.figure(figsize=(15, 10))
plt.ylim(-0.05,0.1)
plt.xlim(-0.05,0.1)
plt.title("Fit sur données d'entraînement", **{'size': 20})
plt.scatter(real, bm, label="Prédiction")
plt.plot(x_base, x_base, c="red", label="Valeur optimale")
plt.legend(fontsize = 15)
plt.xlabel("Valeur réelle", **{'size': 15})
plt.ylabel("Valeur prédite", **{'size': 15});


# COMMAND ----------

bm = model.predict((X_test_mat, X_test_val))


# COMMAND ----------

real = np.transpose([y_test])

x_base = [i/100 for i in list(range(-10, 20))]

plt.figure(figsize=(15, 10))
plt.ylim(-0.05,0.1)
plt.xlim(-0.05,0.1)
plt.title("Fit sur données de test", **{'size': 20})
plt.scatter(real, bm, label="Prédiction")
plt.plot(x_base, x_base, c="red", label="Valeur optimale")
plt.legend(fontsize = 15)
plt.xlabel("Valeur réelle", **{'size': 15})
plt.ylabel("Valeur prédite", **{'size': 15});

# COMMAND ----------

predict = model.predict((X_pred_mat, X_pred_val))

# COMMAND ----------

predict

# COMMAND ----------

anticip = np.transpose([y_pred_arr])

x_base = [i/100 for i in list(range(-10, 20))]

plt.figure(figsize=(15, 10))
plt.ylim(-0.05,0.1)
plt.xlim(-0.05,0.1)
plt.title("Prédictions pour l'année 2022", **{'size': 20})
plt.scatter(anticip, predict, label="Prédiction")
plt.plot(x_base, x_base, c="red", label="Valeur optimale")
plt.legend(fontsize = 15)
plt.xlabel("Valeur anticipée", **{'size': 15})
plt.ylabel("Valeur prédite", **{'size': 15});

# COMMAND ----------

 ncf_diff = pd.DataFrame(predict-anticip, columns = ["ncf_growth differential"], index = labelname).sort_values("ncf_growth differential", ascending = False)

# COMMAND ----------

import plotly.express as px
fig = px.bar(x=ncf_diff.index, y = ncf_diff["ncf_growth differential"].tolist(),
            labels={'x': 'Marché', 'y':'Différentiel du ncf growth en pourcentage'})
fig.update_xaxes(type='category')
fig.write_html("Prediction_file.html")
fig.show()

# COMMAND ----------

ncf__.loc[["New York", "Sacramento", "Kansas City"]][ncf__.columns[18:]]

# COMMAND ----------

city = "Honolulu"

fig, axs = plt.subplots(3, 4)
fig.set_size_inches(40, 12)
fig.suptitle(f"Données géographiques de {city}", **{"size":30})

count = 0

maps_dict2 = {map : maps_dict[map] for map in maps_dict if map not in ["can_landmass", "mex_landmass"]}
for map in maps_dict2:
    
    lon = mkts_ref.loc[mkts_ref.market_publish	== city]["longitude"].item()
    lat = mkts_ref.loc[mkts_ref.market_publish == city]["latitude"].item()
    
    row = count // 3  # integer division
    col = count % 3 
    
    axs[col,row].set_title(map, **{"size":15})
    axs[col,row].imshow(get_zone(lon, lat, map, size = 15), cmap="terrain_r")
    
    count += 1
        

# COMMAND ----------

mkts_ref

# COMMAND ----------

get_zone(lon, lat, (maps_dict[map]), size = 10)

# COMMAND ----------

maps_dict[map]

# COMMAND ----------

lat

# COMMAND ----------

mkts_ref.loc[mkts_ref.market_publish	== city]["longitude"].item()

# COMMAND ----------


