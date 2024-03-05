import os
import json
import requests
import hashlib
import hmac
import base64
import urllib.parse as urlparse
import shutil

from argparse import ArgumentParser
from functools import reduce
from glob import glob
from tqdm import tqdm
from random import randint, seed
from multiprocessing import Pool, current_process


api_key = 'xxxxxxxx' # Your Google Maps Platform API key
secret = 'xxxxxxxx' # Your Google URL signing secret
seed(42)

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

    # original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return encoded_signature.decode()

def make_request(coords, mode="image", size=(560, 560), radius=100, heading=0):

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    if mode == "json":
        base_url += "/metadata"

    url = f"{base_url}?size={size[0]}x{size[1]}&location={coords[0]:.7f},{coords[1]:.7f}&radius={radius}&heading={heading}&source=outdoor&key={api_key}"
    meta_params = {
        'signature': sign_url(url, secret)
    }
    try:
        response = requests.get(url, params=meta_params, timeout=30)
    except:
        return None, 408

    if mode == "json":
        return response.json()

    return response.content, response.status_code

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--locations_dir",
        required=False,
        default="final_locations",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default="geo_dbv2",
        type=str
    )
    return parser.parse_args()

def download_data(county_file, subdivision, output_folder, finished_items):
    
    with open(county_file, "r") as fin:
        county_data = json.load(fin)
    locations = county_data[subdivision]
    county_name = os.path.splitext(os.path.basename(county_file))[0].split("_")[1]
    processed_items = []
    SESSION_CUTOFF = len(locations)
    with tqdm(
        total=SESSION_CUTOFF,
        position=current_process()._identity[0]-1
    ) as pbar:
        pbar.set_description(f"Gathering images for {county_name}, {subdivision}")
        for idx, location in enumerate(locations):
            if location in finished_items:
                continue
            if idx >= SESSION_CUTOFF:
                break
            current_dir = os.path.join(output_folder, county_name, subdivision, f"{idx:04}")
            os.makedirs(current_dir)
            first_heading = randint(0, 89)
            headings = [first_heading, first_heading+90] # first_heading+180, first_heading+270]
            lat, lon = list(map(float, location.split(",")))
            success = 0
            for head_idx, heading in enumerate(headings):
                response_content, status = make_request((lat, lon), heading=heading)
                if status == 200:
                    with open(os.path.join(current_dir, f"{head_idx}.jpg"), "wb") as fout:
                        fout.write(response_content)
                    success += 1
            if success == len(headings):
                processed_items.append(location)
                pbar.update(1)
    return processed_items


def main():

    args = parse_args()
    input_folder = args.locations_dir
    output_folder = os.path.join(args.output_dir, "images")

    os.makedirs(output_folder, exist_ok=True)
    shutil.copy2(os.path.join(input_folder, "locations.json"), args.output_dir)

    files = sorted(glob(os.path.join(input_folder, "metadata_*.json")))
    tasks = []
    if not os.path.exists("finished_items.json"):
        finished_items = [] 
    else:
        with open("finished_items.json", "r") as fin:
            finished_items = json.load(fin)["items"]

    for file in files:
        with open(file, "r") as fin:
            county_data = json.load(fin)
        subdivisions = list(county_data.keys())
        for elem in subdivisions:
            tasks.append((file, elem, output_folder, finished_items))  

    with Pool(processes=10) as pool:
        processed = pool.starmap(download_data, tasks)
        processed_items = reduce(lambda x, y: x+y, processed, [])
        with open("finished_items.json", "w") as fout:
            json.dump({"items": processed_items}, fout, indent=4)


if __name__=="__main__":
    main()