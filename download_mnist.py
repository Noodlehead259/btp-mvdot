import os
import urllib.request
import gzip
import shutil

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
files = [
    ("train-images-idx3-ubyte.gz", "train-images.idx3-ubyte"),
    ("train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte"),
    ("t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte"),
    ("t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte")
]

print("Downloading MNIST dataset...")
for gz_file, extracted_file in files:
    gz_path = os.path.join("data", gz_file)
    extracted_path = os.path.join("data", extracted_file)
    
    if not os.path.exists(extracted_path):
        print(f"Downloading {gz_file}...")
        urllib.request.urlretrieve(base_url + gz_file, gz_path)
        
        print(f"Extracting {gz_file}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(extracted_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Clean up the .gz file
        os.remove(gz_path)
    else:
        print(f"{extracted_file} already exists.")

print("Download complete!")
