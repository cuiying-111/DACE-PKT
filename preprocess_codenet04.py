'''
    这是在load_data.py前面的一步，也是生成用户级数据的最后一步，
    这里实现的是根据code_path从Project_CodeNet.tar。gz中解压相应的代码文本，不过多解压没用的
'''

import os
import json
from tqdm import tqdm

# ===============================
# Config（只改这里）
# ===============================
PROJECT_CODENET_DATA = (
    "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/data"
)

STEP5_JSONL_FILES = [
    "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/AIZU_with_code_seq/train.jsonl",
    "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/AIZU_with_code_seq/valid.jsonl",
    "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/AIZU_with_code_seq/test.jsonl",
]

OUTPUT_CODE_DIR = "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/code_text/AIZU"
LOG_DIR = "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/logs/AIZU"

os.makedirs(OUTPUT_CODE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MISSING_LOG = os.path.join(LOG_DIR, "missing_code_files.log")

# ===============================
# Main
# ===============================
missing = 0
copied = 0

with open(MISSING_LOG, "w") as miss_f:
    for jsonl_path in STEP5_JSONL_FILES:
        print(f"[INFO] Processing {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                rec = json.loads(line)

                for rel_path in rec["code_path_seq"]:
                    # rel_path: data/p02388/C++/s915321514.cpp
                    abs_src = os.path.join(
                        PROJECT_CODENET_DATA,
                        rel_path.replace("data/", "", 1)
                    )

                    if not os.path.isfile(abs_src):
                        missing += 1
                        miss_f.write(abs_src + "\n")
                        continue

                    # 目标路径：code_text/p02388/C++/s915321514.cpp
                    _, pid, lang, fname = rel_path.split("/")
                    target_dir = os.path.join(OUTPUT_CODE_DIR, pid, lang)
                    os.makedirs(target_dir, exist_ok=True)

                    target_path = os.path.join(target_dir, fname)

                    if not os.path.exists(target_path):
                        with open(abs_src, "rb") as fin, open(target_path, "wb") as fout:
                            fout.write(fin.read())
                        copied += 1

print("\n===============================")
print("[Step6] Code collection summary")
print(f"Copied files  : {copied}")
print(f"Missing files : {missing}")
print("===============================\n")

print("[DONE] Step6 finished.")





