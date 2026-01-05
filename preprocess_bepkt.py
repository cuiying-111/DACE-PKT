import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# 现在这个文件对bepkt的处理只是采用普通的KT那种，没有对代码序列做处理，也就是说输入里面还没有code
# ===============================
# Config
# ===============================
SUBMISSION_CSV = "/home/cuiying/projects_paper/DACE-main/data/bepkt/raw_data/submission.csv"
PROBLEM_TAG_CSV = "/home/cuiying/projects_paper/DACE-main/data/bepkt/raw_data/problem_tags.csv"
OUT_DIR = "/home/cuiying/projects_paper/DACE-main/data/bepkt/bepkt_pid"
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 224
import pandas as pd



os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# Load raw data
# ===============================
sub_df = pd.read_csv(SUBMISSION_CSV)
tag_df = pd.read_csv(PROBLEM_TAG_CSV)

print("n_pid =", sub_df["problem_id"].nunique())
print("n_question =", tag_df["problemtag_id"].nunique())

# problem_id -> tag_id
problem2tag = dict(zip(tag_df["problem_id"], tag_df["problemtag_id"]))

# filter submissions without tag
sub_df = sub_df[sub_df["problem_id"].isin(problem2tag)]

# sort by user & time
sub_df = sub_df.sort_values(["user_id", "create_time"])

# ===============================
# Build sequences per student
# ===============================
students = []

for uid, u_df in sub_df.groupby("user_id"):
    pid_seq = []
    qid_seq = []
    a_seq = []

    for _, row in u_df.iterrows():
        pid = int(row["problem_id"]) + 1
        qid = problem2tag[row["problem_id"]] + 1
        a = 1 if row["result"] == 0 else 0  # BePKT: 0 = AC

        pid_seq.append(pid)
        qid_seq.append(qid)
        a_seq.append(a)

    if len(pid_seq) >= 2:
        students.append((pid_seq, qid_seq, a_seq))

print(f"Total students: {len(students)}")

# ===============================
# Train / Valid / Test split
# ===============================
train, temp = train_test_split(
    students, test_size=1 - TRAIN_RATIO, random_state=SEED
)
valid, test = train_test_split(
    temp, test_size=0.5, random_state=SEED
)

# ===============================
# Save in DACE format
# ===============================
def dump(path, data):
    """
    将学生序列写成 DACE / DKVMN 兼容的四行格式：
    第 1 行：样本编号（占位，不参与建模）
    第 2 行：problem id 序列（P）
    第 3 行：knowledge id 序列（Q）
    第 4 行：answer 序列（A）
    """
    with open(path, "w") as f:
        for idx, (pid, qid, a) in enumerate(data):
            # 不要再写死为 0，这里用样本编号更合理
            f.write(str(idx) + "\n")

            f.write(",".join(map(str, pid)) + "\n")
            f.write(",".join(map(str, qid)) + "\n")
            f.write(",".join(map(str, a)) + "\n")

dump(f"{OUT_DIR}/train.txt", train)
dump(f"{OUT_DIR}/valid.txt", valid)
dump(f"{OUT_DIR}/test.txt", test)

print("Preprocess done.")

