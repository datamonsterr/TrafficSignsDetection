# Generate/Download Data
from zipfile import ZipFile

zip_file = input("Path to your dataset: ") 

with ZipFile(zip_file, "r") as zip_data:
    zip_data.extractall("./dataset")
    
print("Complete!")
