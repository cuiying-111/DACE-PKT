'''
    这个脚本实现的功能是，从metadata/pxxxxx.csv中，按照problem_id+编程语言进行筛选，保留以下五种编程语言：C C++ Java JavaScript Python
    生成干净的submission级数据，分别对应两个平台，最终会生成两个数据集： AIZU AtCoder
    注意这一个文件，可以生成两个 数据集的，分别是题目id的,csv文件，
    只需要把下面代码里面的 AIZU换成AtCoder,aizu换成atcoder
    这两个平台互换就可以
'''
import csv
from pathlib import Path

BASE_DIR = Path("/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet")
META_DIR = BASE_DIR / "metadata"
OUT_DIR = BASE_DIR / "processed_metadata" / "AtCoder"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_LANGS = {"C", "C++", "Java", "JavaScript", "Python"}

# 读取 AIZU problem 列表
with open(BASE_DIR / "atcoder_problem_ids.txt") as f:
    aizu_problem_ids = [line.strip() for line in f if line.strip()]

for pid in aizu_problem_ids:
    src_csv = META_DIR / f"{pid}.csv"
    if not src_csv.exists():
        continue

    dst_csv = OUT_DIR / f"{pid}.csv"

    with open(src_csv, newline='', encoding="utf-8") as fin, \
         open(dst_csv, "w", newline='', encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        kept = 0
        for row in reader:
            if row["language"] in ALLOWED_LANGS:
                writer.writerow(row)
                kept += 1

    if kept == 0:
        dst_csv.unlink()  # 没有保留任何记录就删除
