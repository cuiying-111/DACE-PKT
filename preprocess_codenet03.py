'''
    这一步，要做的是，根据之前生成的 AIZU 和 AtCoder两个
    构建按照user的序列级数据，PKT需要，
    直接按照user划分 train / valid / test
    且这里面没有代码的文本内容，是给一个代码的索引
'''

import os
import csv
import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===============================
# Config
# ===============================
DATASET_NAME = "AtCoder"   # or "AIZU"
MAX_USERS = 5000           # None 表示不抽样
RANDOM_SEED = 42
MIN_SEQ_LEN = 20

random.seed(RANDOM_SEED)

BASE_DIR = "/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet"
META_DIR = os.path.join(BASE_DIR, "processed_metadata", DATASET_NAME)
DATA_DIR = os.path.join(BASE_DIR, "data")

OUT_DIR = os.path.join(BASE_DIR, f"{DATASET_NAME}_with_code_seq")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUT_DIR, "train.jsonl")
VALID_FILE = os.path.join(OUT_DIR, "valid.jsonl")
TEST_FILE  = os.path.join(OUT_DIR, "test.jsonl")

TARGET_LANGS = {"C", "C++", "Java", "JavaScript", "Python"}

# ===============================
# Step 5.1 读取 submission → user records
# ===============================
user_records = defaultdict(list)
all_raw_pids = set()

problem_csvs = sorted(
    f for f in os.listdir(META_DIR)
    if f.startswith("p") and f.endswith(".csv")
)

print(f"[{DATASET_NAME}] Problems: {len(problem_csvs)}")

for csv_file in tqdm(problem_csvs, desc="Reading metadata"):
    raw_pid = int(csv_file[1:-4])  # p02388.csv → 2388
    csv_path = os.path.join(META_DIR, csv_file)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["language"] not in TARGET_LANGS:
                continue

            user_id = row["user_id"]
            submission_id = row["submission_id"]
            status = row["status"]
            timestamp = int(row["date"])
            ext = row["filename_ext"]

            a = 1 if status == "Accepted" else 0
            code_path = f"data/{raw_pid}/{row['language']}/{submission_id}.{ext}"

            user_records[user_id].append({
                "time": timestamp,
                "raw_pid": raw_pid,
                "a": a,
                "code_path": code_path
            })

            all_raw_pids.add(raw_pid)

# ===============================
# Step 2: raw_pid → pid_index
# 这里和 BePKT 保持一致，pid 从 1 开始也可以，但要注意 embedding 对齐
# 我用 0-based, 完全安全
# ===============================
raw_pid2pid = {pid: idx for idx, pid in enumerate(sorted(all_raw_pids))}
print(f"[{DATASET_NAME}] n_pid (after re-index) = {len(raw_pid2pid)}")

# ===============================
# Step 3: 构建每个用户的序列
# ===============================
all_users = []

for uid, recs in user_records.items():
    recs = sorted(recs, key=lambda x: x["time"])

    pid_seq = []
    qid_seq = []
    a_seq = []
    code_path_seq = []
    time_seq = []

    for r in recs:
        if r["raw_pid"] not in raw_pid2pid:
            continue
        pid_seq.append(raw_pid2pid[r["raw_pid"]])
        qid_seq.append(raw_pid2pid[r["raw_pid"]])  # BePKT里qid对应知识点，这里用pid_seq
        a_seq.append(r["a"])
        code_path_seq.append(r["code_path"])
        time_seq.append(r["time"])

    if len(pid_seq) < MIN_SEQ_LEN:
        continue

    # 核心检查：pid没有越界
    assert max(pid_seq) < len(raw_pid2pid)

    all_users.append({
        "user_id": uid,
        "pid_seq": pid_seq,
        "qid_seq": qid_seq,
        "a_seq": a_seq,
        "code_path_seq": code_path_seq,
        "time_seq": time_seq
    })

print(f"[{DATASET_NAME}] Valid users: {len(all_users)}")

# ===============================
# Step 5.2.1  用户抽样（可选）
# ===============================
if MAX_USERS is not None and len(all_users) > MAX_USERS:
    all_users = random.sample(all_users, MAX_USERS)
    print(f"[{DATASET_NAME}] Users after sampling: {len(all_users)}")

# ===============================
# Step 5.3  按用户划分数据集
# ===============================
train_recs, temp_recs = train_test_split(
    all_users, test_size=0.2, random_state=RANDOM_SEED
)
valid_recs, test_recs = train_test_split(
    temp_recs, test_size=0.5, random_state=RANDOM_SEED
)

# ===============================
# Step 5.4  写 jsonl
# ===============================
def dump_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump_jsonl(train_recs, TRAIN_FILE)
dump_jsonl(valid_recs, VALID_FILE)
dump_jsonl(test_recs, TEST_FILE)

print("===================================")
print(f"{DATASET_NAME} saved:")
print("Train:", len(train_recs))
print("Valid:", len(valid_recs))
print("Test :", len(test_recs))
print("===================================")
