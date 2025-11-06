import kagglehub
import os

# Download latest version
os.environ['KAGGLEHUB_CACHE'] = "dataset/"
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

print("Path to dataset files:", path)