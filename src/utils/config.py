import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
data_raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
data_uniformed_dir = Path(os.getenv("DATA_UNIFORMED_DIR", "data/uniformed"))
data_cleansed_dir = Path(os.getenv("DATA_CLEANSED_DIR", "data/cleansed"))
data_augmented_dir = Path(os.getenv("DATA_AUGMENTED_DIR", "data/augmented"))

data_categories = {
    1: "伯努利分布",
    2: "贝塔分布",
    3: "二项分布",
    4: "伽马分布",
    5: "高斯混合模型",
    6: "均匀分布",
    7: "卡方分布",
    8: "莱斯分布",
    9: "幂律分布",
    10: "帕累托分布",
    11: "泊松分布",
    12: "正态分布",
    13: "指数分布",
    14: "几何分布",
    15: "狄利克雷分布",
    16: "超几何分布",
}
data_categories_reverse = {v: k for k, v in data_categories.items()}

