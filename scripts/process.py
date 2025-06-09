from utils import data_raw_dir, data_uniformed_dir, data_cleansed_dir, data_augmented_dir
from uniform import uniform
from cleanse import cleanse
from augment import augment

def main():
    # uniform(data_raw_dir, data_uniformed_dir)
    
    # cleanse(data_uniformed_dir, data_cleansed_dir)
    
    augment(data_cleansed_dir, data_augmented_dir)
    

if __name__ == "__main__":
    main() 