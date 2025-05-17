from utils import get_all_files, data_raw_dir, data_cleansed_dir
from collections import Counter
from tqdm import tqdm

files = list(get_all_files(data_raw_dir, True))

# 说明：
# 12_124和12_155只有docx文档，人工将其中图片复制出来
# 14_224y改为16_224y 超几何分布
# 11_93应为rtf，将扩展名由txt改为rtf
def count_file_extension():
    extensions = Counter(file.extension for file in tqdm(files, desc="统计文件扩展名"))
    print(extensions)

# 文件夹的名称格式：类别_序号
# 类别共有13(16)类，分别对应：
# 1. 伯努利分布
# 2. 贝塔分布
# 3. 二项分布
# 4. 伽马分布
# 5. 高斯混合模型
# 6. 均匀分布
# 7. 卡方分布
# 8. 莱斯分布
# 9. 幂律分布
# 10. 帕累托分布
# 11. 泊松分布
# 12. 正态分布
# 13. 指数分布
# // 以下pdf中未给出，从数据推测
# 14. 几何分布
# 15. 狄利克雷分布
# 16. 超几何分布
# 其中序号后带y表示手写
# 统计每个类别的数量
def count_file_category():
    category_counts = {}
    for file in tqdm(list(data_cleansed_dir.iterdir()), desc="统计文件类别"):
        category = int(file.name.split('_')[0])
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1
    print(sorted(category_counts.items(), key=lambda x: x[0], reverse=True))

if __name__ == "__main__":
    count_file_extension()
    count_file_category()