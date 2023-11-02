# Generate/Download Data
from zipfile import ZipFile


def unzip(zip_file, destination):
    with ZipFile(zip_file, "r") as zip_data:
        zip_data.extractall(destination)
