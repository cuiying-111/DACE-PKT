import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# ===============================
# Config
# ===============================
SUBMISSION_CSV = "/home/cuiying/projects_paper/DACE-main/data/bepkt/raw_data/submission.csv"
PROBLEM_TAG_CSV = "/home/cuiying/projects_paper/DACE-main/data/bepkt/raw_data/problem_tags.csv"
OUT_DIR = "/home/cuiying/projects_paper/DACE-main/data/bepkt/bepkt_with_code_seq"
MIN_SEQ_LEN = 2   # 至少多少步才保留一个学生

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUT_DIR, "train.jsonl")
VALID_FILE = os.path.join(OUT_DIR, "valid.jsonl")
TEST_FILE  = os.path.join(OUT_DIR, "test.jsonl")

# ===============================
# Load raw data
# ===============================
sub_df = pd.read_csv(SUBMISSION_CSV)
tag_df = pd.read_csv(PROBLEM_TAG_CSV)

print("Raw submissions:", len(sub_df))

# problem_id -> knowledge_id
problem2tag = dict(zip(tag_df["problem_id"], tag_df["problemtag_id"]))

# 只保留有知识点标注的提交
sub_df = sub_df[sub_df["problem_id"].isin(problem2tag)]

# 按用户 + 时间排序（KT 的生命线）
sub_df = sub_df.sort_values(["user_id", "create_time"])

print("Filtered submissions:", len(sub_df))
print("Unique users:", sub_df["user_id"].nunique())

# ===============================
# Build submission-level sequences
# ===============================
all_records = []
num_students = 0
num_records = 0

for uid, u_df in sub_df.groupby("user_id"):
    pid_seq = []
    qid_seq = []
    a_seq = []
    code_seq = []
    time_seq = []

    for _, row in u_df.iterrows():
        # problem id（从 1 开始，避免 0）
        pid = int(row["problem_id"]) + 1

        # knowledge / tag id
        qid = int(problem2tag[row["problem_id"]]) + 1

        # BePKT: result == 0 表示 AC
        a = 1 if row["result"] == 0 else 0

        # 原始代码文本（暂不做 embedding）
        code_text = str(row["code"]) if not pd.isna(row["code"]) else ""

        # 时间戳（保留，方便以后做时间衰减或分析）
        time_str = str(row["create_time"])

        pid_seq.append(pid)
        qid_seq.append(qid)
        a_seq.append(a)
        code_seq.append(code_text)
        time_seq.append(time_str)

    if len(pid_seq) < MIN_SEQ_LEN:
        continue

    record = {
        "user_id": int(uid),
        "pid_seq": pid_seq,     # 题目序列（submission-level）
        "qid_seq": qid_seq,     # 知识点序列
        "a_seq": a_seq,         # 作答结果序列
        "code_seq": code_seq,   # 代码文本序列
        "time_seq": time_seq    # 时间序列
    }

    all_records.append(record)
    num_students += 1
    num_records += len(pid_seq)

print("===================================")
print(f"Total valid students: {num_students}")
print(f"Total interactions: {num_records}")

# ===============================
# Train / Valid / Test split
# （按学生划分，而不是按提交）
# ===============================
train_recs, temp_recs = train_test_split(
    all_records, test_size=0.2, random_state=42
)

valid_recs, test_recs = train_test_split(
    temp_recs, test_size=0.5, random_state=42
)

def dump_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump_jsonl(train_recs, TRAIN_FILE)
dump_jsonl(valid_recs, VALID_FILE)
dump_jsonl(test_recs, TEST_FILE)

print("===================================")
print("Saved files:")
print(" -", TRAIN_FILE, f"({len(train_recs)} students)")
print(" -", VALID_FILE, f"({len(valid_recs)} students)")
print(" -", TEST_FILE,  f"({len(test_recs)} students)")
print("Preprocess DONE.")
