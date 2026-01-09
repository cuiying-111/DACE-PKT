'''
这是对Project_CodeNet数据集处理的第一步，这里面是先处理problem_list.csv文件，先分出来两个平台的问题 id
分别是AIZU 和 AtCoder两个数据集
先生成的是两个.txt文件，只读取 problem_list.csv这一个文件即可，
运行的结果如下：
AIZU problems: 2534
AtCoder problems: 1519
'''

import csv
from pathlib import Path

base_dir = Path("/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet")
problem_list_path = base_dir / "metadata/problem_list.csv"

aizu_ids = []
atcoder_ids = []

with open(problem_list_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["id"]
        dataset = row["dataset"]

        if dataset == "AIZU":
            aizu_ids.append(pid)
        elif dataset == "AtCoder":
            atcoder_ids.append(pid)

# 写出索引文件
with open(base_dir / "aizu_problem_ids.txt", "w") as f:
    f.write("\n".join(aizu_ids))

with open(base_dir / "atcoder_problem_ids.txt", "w") as f:
    f.write("\n".join(atcoder_ids))

print(f"AIZU problems: {len(aizu_ids)}")
print(f"AtCoder problems: {len(atcoder_ids)}")
