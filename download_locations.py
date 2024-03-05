import os
import geopandas as gpd
import pandas as pd
import json
import matplotlib.pyplot as plt
import requests
import hashlib
import hmac
import base64
import urllib.parse as urlparse
from glob import glob
from difflib import get_close_matches
from unidecode import unidecode
from tqdm import tqdm
from multiprocessing import Pool, current_process


api_key = 'xxxxxxxx' # Your Google Maps Platform API key
secret = 'xxxxxxxx' # Your Google URL signing secret

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signature
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return encoded_signature.decode()

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

def make_request(pt, radius):

    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?size=600x400&location={pt.y:.7f},{pt.x:.7f}&radius={radius}&source=outdoor&key={api_key}"
    meta_params = {
        'signature': sign_url(url, secret)
    }
    meta_response = requests.get(url, params=meta_params)
    res = meta_response.json()
    return res

def process_func(county_name, df3, merged, top_cities):

    CNT_CITY_TOTAL = 500
    CNT_INTERCITY = 200
    county_outline = df3[df3.NAME_1==county_name].geometry
    metadata = {}
    
    skipped = 0
    used = set()
    
    if county_name != "Bucuresti":
        
        city_count = len(top_cities[county_name].index)
        if county_name in ["Giurgiu", "Dolj"]:
            city_count = 3
        city_quota = CNT_CITY_TOTAL // city_count

        for i, city_idx in enumerate(top_cities[county_name].index):
            # fig, ax = plt.subplots()
            if (county_name in ["Giurgiu", "Dolj"]) and i == 3:
                break
            city_name = merged.iloc[city_idx]["city"]
            metadata[city_name] = []
            sample_size = city_quota if i > 0 else (city_quota + (CNT_CITY_TOTAL % city_count))
            city_outline = merged[merged.index == city_idx]
            co = merged.iloc[city_idx]["geometry"]
            lng = merged.iloc[city_idx]["lng"]
            lat = merged.iloc[city_idx]["lat"]
            
            points = city_outline.sample_points(
                method="normal",
                center=(lng, lat),
                cov=0.001,
                size=5000
            )
            pts = points.explode(ignore_index=True, index_parts=False).sample(frac=1).reset_index(drop=True)
            pb = tqdm(
                total=sample_size,
                desc=f"Gathering points for {county_name}, {city_name}",
                position=current_process()._identity[0]-1
            )
            for pt in pts.values:
                try:
                    res = make_request(pt, radius=1000)
                    if (res["status"] == "OK") and ("Google" in res["copyright"]):
                        loc = ",".join([f"{res['location'][k]:.8f}" for k in ["lat","lng"]])
                        if loc not in used:
                            used.add(loc)
                            metadata[city_name].append(loc)
                            pb.update(1)
                            if len(metadata[city_name]) == sample_size:
                                break
                except:
                    skipped += 1
            pb.close()

            # city_outline.plot(ax=ax, color="white", edgecolor="black")
            # points.plot(ax=ax, marker='o', color='red', markersize=5)
            # plt.show()
            county_outline = county_outline.difference(co)
        
    # fig, ax = plt.subplots()
    k = "intercity" if county_name != "Bucuresti" else "Bucuresti"
    r = 1000 # 500 if county_name != "Bucuresti" else 300
    metadata[k] = []
    points = county_outline.sample_points(
        size=10000
    )
    intercity_points = points.explode(ignore_index=True, index_parts=False).sample(frac=1).reset_index(drop=True)
    pb = tqdm(
        total=CNT_INTERCITY,
        desc=f"Gathering points for {county_name}, {k}",
        position=current_process()._identity[0]-1
    )
    for pt in intercity_points.values:
        try:
            res = make_request(pt, radius=r)
            if (res["status"] == "OK") and ("Google" in res["copyright"]):
                loc = ",".join([f"{res['location'][k]:.8f}" for k in ["lat","lng"]])
                if loc not in used:
                    used.add(loc)
                    metadata[k].append(loc)
                    pb.update(1)
                    limit = CNT_INTERCITY if county_name != "Bucuresti" else (CNT_INTERCITY + CNT_CITY_TOTAL)
                    if len(metadata[k]) == limit:
                        break
        except:
            skipped+=1

    pb.close()
    with open(f"final_locations/metadata_{county_name}.json", "w") as fout:
        json.dump(metadata, fout, ensure_ascii=True, indent=4)

if __name__=="__main__":

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
    top_cities = merged.groupby("admin_name")["population"].nlargest(4)

    os.makedirs("final_locations", exist_ok=True)
    with Pool(processes=10) as pool:
        pool.starmap(process_func, [(elem, df3, merged, top_cities) for elem in list(sorted(set(df3.NAME_1)))])


    files = glob("final_locations/metadata_*.json")
    d = {}
    for file in files:
        cname = os.path.basename(file).split("_")[1].split(".")[0]
        with open(file, "r") as fin:
            data = json.load(fin)
        d[cname] = data

    with open("final_locations/locations.json", "w") as fout:
        json.dump(d, fout, indent=4)

    # county_outline.plot(ax=ax, color="white", edgecolor="black")
    # points.plot(ax=ax, marker='o', color='red', markersize=5)
    # plt.show()

