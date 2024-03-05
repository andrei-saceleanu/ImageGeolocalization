import sys
import geopandas as gpd
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from difflib import get_close_matches
from unidecode import unidecode
from shapely import Point

def func(x):

    res = None
    if " " not in x and "-" not in x:
        res = x
    
    if "-" in x:
        parts = x.split("-")
        if " " not in parts[1]:
            res = parts[0] + "-" + parts[1].lower()
        else:
            p2 = parts[1].split(" ")
            res =  parts[0] + "-" + p2[0].lower() + "".join([elem.title() for elem in p2[1:]])
    else:
        res = x.title().replace(" ","")
    if res not in df1.NAME_2:
        m = get_close_matches(res, df1.NAME_2)
        if m:
            res = m[0]
    return res


input_dir = sys.argv[1]
county_to_display = sys.argv[2]

df1 = gpd.read_file("gadm41_ROU_2.json")
df2 = pd.read_excel("ro_cities.xlsx")
df3 = gpd.read_file("gadm41_ROU_1.json")


df1.loc[df1.NAME_1=="Bucharest", "NAME_1"] = "Bucuresti"
df1.loc[df1.NAME_1=="Bucuresti", "NAME_2"] = "Bucuresti"
df2.loc[df2.city=="Bucharest","city"] = "Bucuresti"

df1.NAME_1 = df1.NAME_1.apply(lambda x: unidecode(x))
df1.NAME_2 = df1.NAME_2.apply(lambda x: unidecode(x))

df2.city = df2.city.apply(lambda x: unidecode(x))
df2.admin_name = df2.admin_name.apply(lambda x: unidecode(x).replace(" ",""))
df2.city = df2.city.apply(lambda x:func(x))

df3.loc[df3.NAME_1=="Bucharest","NAME_1"] = "Bucuresti"
df3.NAME_1 = df3.NAME_1.apply(lambda x: unidecode(x))

merged = df1.merge(
    df2,
    left_on=["NAME_2", "NAME_1"],
    right_on=["city", "admin_name"],
    how="outer"
)
merged = merged[merged.NAME_2.notna() & merged.city.notna()].drop_duplicates(subset=["NAME_2", "NAME_1"],ignore_index=True)

with open(os.path.join(input_dir, "locations.json"), "r") as fin:
    coords_data = json.load(fin)

county = county_to_display
city_outline = df3[df3.NAME_1==county]

l = []
for k in coords_data[county]:
    coords = coords_data[county][k]
    l.extend([Point(*list(map(float,elem.split(",")))[::-1])for elem in coords])
points = gpd.GeoSeries(l)

fig, ax = plt.subplots()
city_outline.plot(ax=ax, color="white", edgecolor="black")
points.plot(ax=ax, marker='o', color='red', markersize=5)
plt.show()
