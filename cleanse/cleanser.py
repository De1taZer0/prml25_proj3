from PIL import Image
import re
from utils.file import FilePath
from pathlib import Path
from tqdm import tqdm
import sys
import os
import asyncio
from typing import List

from utils import get_all_files, get_output_path, process_files_async
from .processors import create_processor

def cleanse_file(input_path: FilePath, output_path: FilePath) -> None:
    """根据文件类型调用相应的处理函数"""
    processor = create_processor(input_path)
    processor.process(input_path, output_path)

def cleanse(data_uniformed_dir: Path, data_cleansed_dir: Path):
    """清洗处理目录中的所有文件"""
    for dir in tqdm(data_uniformed_dir.iterdir(), desc="清洗目录", total=len(os.listdir(data_uniformed_dir)), position=0, file=sys.stdout):
        assert dir.is_dir(), f"清洗目录中存在非目录文件: {dir}"
        files = list(get_all_files(dir, True))
        asyncio.run(process_files_async(files, data_cleansed_dir, dir.name, cleanse_file, "清洗"))