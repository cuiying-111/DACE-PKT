import json
import os

DATA_DIR = "/home/cuiying/projects_paper/DACE-main/data/csedm_f/f19_with_code_seq"
FILES = ["train.jsonl", "valid.jsonl", "test.jsonl"]

pid_set = set()
qid_set = set()
num_students = 0
num_interactions = 0

for fname in FILES:
    path = os.path.join(DATA_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            num_students += 1

            pid_seq = record["pid_seq"]
            qid_seq = record["qid_seq"]

            pid_set.update(pid_seq)
            qid_set.update(qid_seq)

            num_interactions += len(pid_seq)

print("===================================")
print("Dataset dir:", DATA_DIR)
print("Total students      :", num_students)
print("Total interactions  :", num_interactions)
print("Unique pid (n_pid)  :", len(pid_set))
print("Unique qid (n_question):", len(qid_set))
print("pid min / max:", min(pid_set), max(pid_set))
print("qid min / max:", min(qid_set), max(qid_set))
print("===================================")
