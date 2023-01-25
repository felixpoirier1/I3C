from flask import Flask, render_template
from flask_sock import Sock
import folium
from bs4 import BeautifulSoup
import pandas as pd
app = Flask(__name__, static_folder='static')
sock = Sock(app)



@app.route("/")
def home():
    context = {}
    context["cities"] = ["New York", "Kansas City", "San Francisco"]
    context["citiesfilename"] = ["newyork", "kansascity", "sanfrancisco"]
    return render_template("home.html", context=context)

@app.route(f"/newyork")
def newyork():
    city = "newyork"
    name = "New York"
    context = {}
    with open(f"maps_html/newyork.html", "r") as f:
        context["map"] = f.read()
    
    return render_template(f"{city}.html", context=context)

@app.route(f"/kansascity")
def kansascity():
    city = "kansascity"
    name = "Kansas City"
    context = {}
    with open(f"maps_html/kansascity.html", "r") as f:
        context["map"] = f.read()
    return render_template(f"{city}.html", context = context)

@app.route(f"/sanfrancisco")
def sanfrancisco():
    city = "sanfrancisco"
    name = "San Francisco"
    context = {}
    with open(f"maps_html/sanfrancisco.html", "r") as f:
        context["map"] = f.read()
    return render_template(f"{city}.html", context=context)

@app.route(f"/allcities")
def allcities():
    m = folium.Map(location=[38.482845, -97.494392], zoom_start=5, tiles="cartodbpositron")
    cities = pd.read_csv("american_cities.csv").drop(columns=["Unnamed: 0"])
    cities = cities.drop_duplicates(keep="first", subset=["city"])
    cities.set_index("city", inplace=True)
    for city in cities.index:
        lat = cities.loc[city, "lat"]
        lon = cities.loc[city, "lng"]
        folium.Circle([lat, lon], popup=city).add_to(m)
    
    html_doc = m.get_root().render()
    soup = BeautifulSoup(html_doc, 'html.parser')
    head = str(soup.head)[6:-8]
    body = str(soup.body)[6:-8]
    script = soup.find_all("script")
    return render_template("allcities.html", context = {"head": head, "map": body, "script": script})