import geopandas as gpd
import matplotlib.pyplot as plt
from glob import glob
import json
from pprint import pprint

processed = glob("metadata_*.json")
proc_locs = []
for elem in processed:
    with open(elem, "r") as fin:
        curr_data = json.load(fin)
    for k, v in curr_data.items():
        proc_locs.extend([(elem,k) for elem in v])

d = {}
for elem in proc_locs:
    if elem[0] not in d:
        d[elem[0]] = [elem[1]]
    else:
        d[elem[0]].append(elem[1])
pprint({k:set(v) for k,v in d.items() if len(v) > 1})
# print(len(proc_locs))
# used = set(proc_locs)
# print(len(used))
# df1 = gpd.read_file("gadm41_ROU_2.json")

# city = df1[df1.NAME_2=="MunicipiulBucuresti"]

# lat = 44.4325
# lng = 26.1039
# fig, ax = plt.subplots()

# city.plot(ax=ax, color="white", edgecolor="black")
# points = city.sample_points(
#     method="normal",
#     center=(lng, lat),
#     cov=0.001,
#     size=5000
# )
# pts = points.explode(ignore_index=True, index_parts=False).sample(frac=1).reset_index(drop=True).iloc[:200]

# pts.plot(ax=ax, marker='o', color='red', markersize=5)
# plt.scatter(x=lng, y=lat, color="green", marker="o")
# plt.show()