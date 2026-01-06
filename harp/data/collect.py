import urllib.request
import os
from tqdm import tqdm
import shutil
import glob
import pandas as pd


# get the sample files from the cms synthetic data and put into data/raw/samples
def retrieve_data(total_samples=5, save_dir="data/raw/samples"):
     if total_samples > 20:
          raise ValueError(f"Invalid Number of Samples {total_samples}; Must be less than or equal to 20.")
     sample_range = range(1, total_samples+1)

     full_save_dir = os.path.join("harp", save_dir)
     # print(full_save_dir)
     os.makedirs(full_save_dir, exist_ok=True)

     for sample in tqdm(sample_range, "Downloading Samples"):
          dnld_format = f"https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads/de1_0_2008_to_2010_inpatient_claims_sample_{sample}.zip"
          save_path = os.path.join("harp", save_dir, f"sample_{sample}.zip")
          # print(save_path)
          urllib.request.urlretrieve(dnld_format, save_path)
          shutil.unpack_archive(save_path, os.path.join("harp", save_dir))
          os.remove(save_path)


# process the samples into one joined csv
def combine_csv(sample_dir="data/raw/samples"):

     save_dir = os.path.join("harp", "data", "raw", "cms_2008_2010_samples.csv")

     # print(save_dir)
     # print(os.path.join(harp_dir, "harp", sample_dir, "**", "*.csv"))

     files = glob.glob(os.path.join("harp", sample_dir, "**", "*.csv"), recursive=True)
     pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(save_dir, index=False)
     

#download all raw data + combine into one csv
def collect(total_samples, save_dir):
     retrieve_data(total_samples, save_dir)
     combine_csv(save_dir)