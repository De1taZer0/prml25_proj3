from utils import data_raw_dir, data_uniformed_dir, data_cleansed_dir
from uniform import uniform
from cleanse import cleanse

def main():
    uniform(data_raw_dir, data_uniformed_dir)

    cleanse(data_uniformed_dir, data_cleansed_dir)

if __name__ == "__main__":
    main() 