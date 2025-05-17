import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
data_uniformed_dir = Path(os.getenv("DATA_UNIFORMED_DIR", "data/uniformed"))
data_cleansed_dir = Path(os.getenv("DATA_CLEANSED_DIR", "data/cleansed"))


