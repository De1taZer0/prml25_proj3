from tqdm import tqdm
from utils.file import FilePath
from utils.word import cleanup_word_app
from utils import process_files_async, get_all_files
from pathlib import Path
import sys
import os
import asyncio
from .processors import create_processor

def uniform_file(input_path: FilePath, output_path: FilePath) -> None:
    """根据文件类型调用相应的处理函数"""
    processor = create_processor(input_path)
    processor.process(input_path, output_path)

def uniform(data_raw_dir: Path, data_uniformed_dir: Path):
    """统一处理目录中的所有文件"""
    try:
        for dir in tqdm(data_raw_dir.iterdir(), desc="处理目录", total=len(os.listdir(data_raw_dir)), position=0, file=sys.stdout):
            assert dir.is_dir(), f"原始数据目录中存在非目录文件: {dir}"
            files = list(get_all_files(dir, True))
            asyncio.run(process_files_async(files, data_uniformed_dir, dir.name, uniform_file, "统一"))
    except Exception as e:
        tqdm.write(f"处理目录 {data_raw_dir} 时发生错误: {e}")
        raise e
    finally:
        cleanup_word_app()
