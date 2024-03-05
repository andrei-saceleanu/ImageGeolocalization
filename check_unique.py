from glob import glob
import os
import json
from collections import Counter

input_folder = "final_locations"

files = sorted(glob(os.path.join(input_folder, "metadata_*.json")))
l = []
for file in files:
    with open(file, "r") as fin:
        county_data = json.load(fin)
    for k, v in county_data.items():
        l.extend(v)
print(len(l))
print(len(set(l)))
c = Counter(l).most_common(2)
print(c)

for file in files:
    with open(file, "r") as fin:
        county_data = json.load(fin)
    for k, v in county_data.items():
        # l.extend(v)
        if c[0][0] in v:
            print(c[0][0])
            print(file)
            print(k)
        if c[1][0] in v:
            print(c[1][0])
            print(file) 
            print(k)
            