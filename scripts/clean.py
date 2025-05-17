import os
import shutil
from pathlib import Path

def main():
    """清理数据目录"""
    print("正在清理数据目录...")
    
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    
    # 检查目录是否存在
    if not data_dir.exists():
        print("错误：data目录不存在！")
        return
    
    if not raw_dir.exists():
        print("错误：data/raw目录不存在！")
        return
    
    # 删除除raw外的所有目录
    for item in data_dir.iterdir():
        if item.is_dir() and item != raw_dir:
            print(f"正在删除目录: {item}")
            shutil.rmtree(item)
    
    print("清理完成！")

if __name__ == "__main__":
    main() 