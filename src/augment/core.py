from pathlib import Path
import os
import re
from collections import Counter
from tqdm import tqdm
import sys
import asyncio
from typing import Dict, List, Tuple

from utils import get_all_files, get_output_path, process_files_async, FilePath
from .processors import create_processor

def parse_category_name(dir_name: str) -> Tuple[str, bool]:
    """解析类别名称，返回(类别编号, 是否手写)"""
    match = re.match(r"(\d+)(?:_\d+)?(y)?", dir_name)
    if not match:
        raise ValueError(f"无效的目录名称格式: {dir_name}")
    return match.group(1), bool(match.group(2))

def generate_new_category_name(category: str, is_handwritten: bool, index: int) -> str:
    """生成新的类别名称"""
    return f"{category}_{index}{'y' if is_handwritten else ''}"

def count_samples_by_category(data_dir: Path) -> Dict[str, Dict[bool, int]]:
    """统计每个类别的样本数量，区分手写和非手写
    
    Returns:
        Dict[str, Dict[bool, int]]: {类别: {是否手写: 数量}}
    """
    sample_counts = {}
    for dir in data_dir.iterdir():
        if dir.is_dir():
            category, is_handwritten = parse_category_name(dir.name)
            if category not in sample_counts:
                sample_counts[category] = {True: 0, False: 0}
            sample_counts[category][is_handwritten] += 1
    return sample_counts

def calculate_augmentation_targets(sample_counts: Dict[str, Dict[bool, int]]) -> Dict[str, Dict[bool, int]]:
    """计算每个类别需要增强的数量，区分手写和非手写
    
    Returns:
        Dict[str, Dict[bool, int]]: {类别: {是否手写: 目标数量}}
        注意：返回的是总目标数量，而不是需要增加的数量
    """
    # 找出手写和非手写分别的最大数量
    max_counts = {
        True: max((counts[True] for counts in sample_counts.values()), default=0),
        False: max((counts[False] for counts in sample_counts.values()), default=0)
    }
    
    augmentation_targets = {}
    for category, counts in sample_counts.items():
        augmentation_targets[category] = {
            True: max(max_counts[True], counts[True]),  # 使用最大值，保证数量足够的类别也会增强
            False: max(max_counts[False], counts[False])
        }
    
    return augmentation_targets

async def augment_category_dir(
    input_dir: Path,
    output_dir: Path,
    source_dir: Path,
    target_count: int
) -> None:
    """对单个类别目录进行增强"""
    category, is_handwritten = parse_category_name(source_dir.name)
    
    # 复制原始文件
    target_dir = output_dir / source_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for file in source_dir.iterdir():
        if file.name in ['description.txt', 'formula.png']:
            output_path = target_dir / file.name
            processor = create_processor(file)
            processor.process(file, output_path)
    
    # 如果不需要增强，直接返回
    if target_count <= 0:
        return
    
    # 进行数据增强
    description_file = source_dir / 'description.txt'
    formula_file = source_dir / 'formula.png'
    
    if not (description_file.exists() and formula_file.exists()):
        tqdm.write(f"警告: {source_dir} 目录下缺少必要的文件")
        return
    
    # 从1001开始编号
    current_index = 1001
    
    # 创建增强后的文件
    text_processor = create_processor(description_file)
    image_processor = create_processor(formula_file)
    
    # 获取所有可能的图像增强结果
    temp_dir = Path("temp_augment")
    temp_dir.mkdir(exist_ok=True)
    image_paths = image_processor.augment(formula_file, temp_dir)
    
    # 过滤出所有不同的增强结果（排除formula.png）
    augmented_images = [path for path in image_paths if path.name != 'formula.png']
    
    # 如果增强结果不够，需要重复使用
    while len(augmented_images) < target_count:
        augmented_images.extend([path for path in image_paths if path.name != 'formula.png'])
    
    with tqdm(total=target_count, desc=f"增强{category}({'手写' if is_handwritten else '非手写'})", position=1, leave=False) as pbar:
        for _ in range(target_count):
            # 为每个新样本创建一个新目录
            new_dir_name = generate_new_category_name(category, is_handwritten, current_index)
            new_dir = output_dir / new_dir_name
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # 增强文本
            text_processor.augment(description_file, new_dir)
            
            # 选择一个增强后的图像
            selected_image = augmented_images[_ % len(augmented_images)]
            output_path = new_dir / 'formula.png'
            import shutil
            shutil.copy2(selected_image, output_path)
            
            current_index += 1
            pbar.update(1)
    
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

def augment(data_cleansed_dir: Path, data_augmented_dir: Path):
    """对数据集进行增强，平衡各个类别的样本数量"""
    try:
        # 1. 统计每个类别的样本数量（区分手写和非手写）
        sample_counts = count_samples_by_category(data_cleansed_dir)
        tqdm.write("各类别样本数量统计:")
        for category, counts in sample_counts.items():
            tqdm.write(f"类别 {category}:")
            tqdm.write(f"  - 手写: {counts[True]}")
            tqdm.write(f"  - 非手写: {counts[False]}")
        
        # 2. 计算目标数量
        augmentation_targets = calculate_augmentation_targets(sample_counts)
        
        tqdm.write("\n增强后的目标数量:")
        for category, counts in augmentation_targets.items():
            if counts[True] > 0:
                current = sample_counts[category][True]
                target = counts[True]
                tqdm.write(f"类别 {category} - 手写: {current} -> {target} 个样本")
            if counts[False] > 0:
                current = sample_counts[category][False]
                target = counts[False]
                tqdm.write(f"类别 {category} - 非手写: {current} -> {target} 个样本")
        
        # 3. 对每个类别进行增强
        for dir in tqdm(list(data_cleansed_dir.iterdir()), desc="处理类别", position=0):
            if not dir.is_dir():
                continue
                
            category, is_handwritten = parse_category_name(dir.name)
            target = augmentation_targets.get(category, {}).get(is_handwritten, 0)
            
            # 所有类别都进行增强
            asyncio.run(augment_category_dir(
                data_cleansed_dir,
                data_augmented_dir,
                dir,
                target  # 使用目标数量
            ))
            
    except Exception as e:
        tqdm.write(f"数据增强过程中发生错误: {e}")
        raise e 