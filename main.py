# 预处理数据

import os
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm

from utils import get_all_files
from uniform import uniform
from cleanse import cleanse

def main():
    data_raw_dir = Path("data/raw")
    data_uniformed_dir = Path("data/uniformed")
    data_cleansed_dir = Path("data/cleansed")
    
    files = list(get_all_files(data_raw_dir, True))
    extensions = Counter(file.extension for file in tqdm(files, desc="统计文件扩展名"))
    print(extensions)

    # 说明：
    # 12_124和12_155只有docx文档，人工将其中图片复制出来
    
    uniform(data_raw_dir, data_uniformed_dir)

    cleanse(data_uniformed_dir, data_cleansed_dir)

if __name__ == "__main__":
    main() 