import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import requests
import tarfile
from tqdm import tqdm
from nilearn import plotting, datasets

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_parser import parse_basic_params

# Load basic parameters from config
basic_parameters = parse_basic_params()
HCP_DIR = basic_parameters['HCP_DIR']
N_SUBJECTS = basic_parameters['N_SUBJECTS']
N_PARCELS = basic_parameters['N_PARCELS']

if not os.path.isdir(HCP_DIR):
    os.makedirs(HCP_DIR, exist_ok=True)

# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in sec

# Argument parsing
import argparse
parser = argparse.ArgumentParser(description='Download HCP Data')
parser.add_argument('--dry-run', action='store_true', help='Check if data can be downloaded without actually downloading')
args = parser.parse_args()

## Downloading data
fnames = ["hcp_rest.tgz",
          "hcp_task.tgz",
          "hcp_covariates.tgz",
          "atlas.npz"]
urls = ["https://osf.io/bqp7m/download",
        "https://osf.io/s4h8j/download",
        "https://osf.io/x5p4g/download",
        "https://osf.io/j5kuc/download"]

for fname, url in zip(fnames, urls):
    if args.dry_run:
        print(f"[Dry Run] would check/download {fname} from {url}")
        try:
            response = requests.head(url, allow_redirects=True)
            if response.status_code == 200:
                print(f"  Connection to {url} successful (Status 200)")
            else:
                print(f"  Connection to {url} failed (Status {response.status_code})")
        except requests.ConnectionError:
             print(f"  Connection to {url} failed (ConnectionError)")
        continue

    # Download to HCP_DIR
    target_path = os.path.join(HCP_DIR, fname)
    if not os.path.isfile(target_path) or os.path.getsize(target_path) == 0:
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                total_size = int(r.headers.get('content-length', 0))
                print(f"Downloading {fname} ({total_size/(1024*1024):.2f} MB)...")
                with open(target_path, "wb") as fid, tqdm(
                    desc=fname,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = fid.write(chunk)
                        bar.update(size)
                print(f"Download {fname} completed!")
            else:
                print(f"!!! Failed to download {fname} - status {r.status_code} !!!")
        except requests.ConnectionError:
            print(f"!!! Connection Error downloading {fname} !!!")

if args.dry_run:
    sys.exit(0)

# Extraction
tgz_files = ["hcp_covariates", "hcp_rest", "hcp_task"]
for fname in tgz_files:
    target_dir = os.path.join(HCP_DIR, fname)
    if not os.path.exists(target_dir):
        tgz_path = os.path.join(HCP_DIR, f"{fname}.tgz")
        if os.path.exists(tgz_path):
            print(f"Extracting {fname}.tgz...")
            with tarfile.open(tgz_path) as fzip:
                fzip.extractall(HCP_DIR)
        else:
            print(f"Warning: {tgz_path} not found for extraction.")
    else:
        print(f"Directory {target_dir} already exists.")

## Loading region information for verification
regions_path = os.path.join(HCP_DIR, "hcp_task", "regions.npy")
if os.path.exists(regions_path):
    regions = np.load(regions_path).T
    region_info = dict(name=regions[0].tolist(),
                       network=regions[1],
                       myelin=regions[2].astype(float))
    print("Region information loaded successfully.")

atlas_path = os.path.join(HCP_DIR, "atlas.npz")
if os.path.exists(atlas_path):
    with np.load(atlas_path) as dobj:
        atlas = dict(**dobj)
    print("Atlas loaded successfully.")
