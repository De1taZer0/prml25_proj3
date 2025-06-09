import asyncio
from typing import List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys

from .file import FilePath, get_output_path

async def process_files_async(
    files: List[FilePath],
    output_dir: Path,
    dir_name: str,
    process_func: Callable[[FilePath, FilePath], None],
    desc: str = "处理"
) -> None:
    """异步处理文件
    
    Args:
        files: 要处理的文件列表
        output_dir: 输出目录
        dir_name: 目录名称（用于显示进度）
        process_func: 处理函数
        desc: 进度条描述
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        tasks = []
        for file in files:
            if not file.valid:
                continue
            output_path = get_output_path(file, output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            task = asyncio.get_event_loop().run_in_executor(
                executor,
                process_func,
                file,
                output_path
            )
            tasks.append(task)

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{desc} {dir_name}", leave=False, position=1):
            await task 