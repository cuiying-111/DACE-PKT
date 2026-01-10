'''
    这个文件实现的是对CSEDM这个数据集的处理，这个数据集让我分成了春季和秋季的
    分别为csedm_f和csedm_s,也就是F19和S19，
    这个文件下面的脚本，只需要替换地址和名称就可以换数据集，CSEDM_F19和CSEDM_S19
'''
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

# ===============================
# Config
# ===============================
DATA_DIR = "/home/cuiying/projects_paper/DACE-main/data/csedm_f/All/Data"
OUT_DIR  = "/home/cuiying/projects_paper/DACE-main/data/csedm_f/f19_with_code_seq"
MIN_SEQ_LEN = 2

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(OUT_DIR, "train.jsonl")
VALID_FILE = os.path.join(OUT_DIR, "valid.jsonl")
TEST_FILE  = os.path.join(OUT_DIR, "test.jsonl")

# ===============================
# Load raw tables
# ===============================
main_df = pd.read_csv(os.path.join(DATA_DIR, "MainTable.csv"))
code_df = pd.read_csv(os.path.join(DATA_DIR, "CodeStates", "CodeStates.csv"))

print("Raw MainTable rows:", len(main_df))

# ===============================
# 1. 只保留 Run.Program（一次有效提交）
# ===============================
main_df = main_df[main_df["EventType"] == "Run.Program"]

print("Run.Program rows:", len(main_df))

# ===============================
# 2. 排序（KT 的生命线）
# ===============================
main_df = main_df.sort_values(
    ["SubjectID", "AssignmentID", "ProblemID", "Order"]
)

# ===============================
# 3. 构建 ProblemID -> qid（problem-as-skill）
# ===============================
unique_problems = sorted(main_df["ProblemID"].unique())
problem2qid = {pid: idx + 1 for idx, pid in enumerate(unique_problems)}

print("Total unique problems (skills):", len(problem2qid))

# ===============================
# 4. CodeStateID -> Code 映射
# ===============================
code_map = dict(zip(code_df["CodeStateID"], code_df["Code"]))

# ===============================
# Build student-level sequences
# ===============================
all_records = []
num_students = 0
num_interactions = 0

for sid, u_df in main_df.groupby("SubjectID"):
    pid_seq = []
    qid_seq = []
    a_seq = []
    code_seq = []
    time_seq = []

    for _, row in u_df.iterrows():
        # problem id（从 1 开始）
        #pid = int(row["ProblemID"]) + 1
        pid = problem2qid[row["ProblemID"]]

        # knowledge id（problem-as-skill）
        qid = problem2qid[row["ProblemID"]]

        # score -> binary correctness
        score = row["Score"]
        a = 1 if pd.notna(score) and score > 0 else 0

        # code text
        code = code_map.get(row["CodeStateID"], "")
        code = str(code) if pd.notna(code) else ""

        # 时间（保留原始字符串）
        time_str = str(row["ServerTimestamp"])

        pid_seq.append(pid)
        qid_seq.append(qid)
        a_seq.append(a)
        code_seq.append(code)
        time_seq.append(time_str)

    if len(pid_seq) < MIN_SEQ_LEN:
        continue

    record = {
        "user_id": sid,
        "pid_seq": pid_seq,
        "qid_seq": qid_seq,
        "a_seq": a_seq,
        "code_seq": code_seq,
        "time_seq": time_seq
    }

    all_records.append(record)
    num_students += 1
    num_interactions += len(pid_seq)

print("===================================")
print(f"Total valid students: {num_students}")
print(f"Total interactions : {num_interactions}")

# ===============================
# Train / Valid / Test split
# （按学生）
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
