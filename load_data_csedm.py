import json
import math
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
def select_bias(qa_array, num, bias_type="None", bias_p=0.0, select_pro=None):
    # 根据指定的异常行为类型（抄袭 / 瞎猜），对单个学生的作答序列进行噪声注入，修改部分作答结果为异常值。
    if bias_type == "plag":
        # 噪声类型：抄袭（模拟学生抄袭答案，强行把部分作答改成对）
        # 1. 随机选择要注入噪声的位置：从0~num-1中选 int(num*bias_p) 个索引
        select_idx = np.random.permutation(range(num))[0: int(num * bias_p)]
        qa_array[select_idx] = 1
    elif bias_type == "plag_by_pro":
        # 噪声类型：按题目抄袭（只对指定题目抄袭做对）
        # 1. 判断前num个作答中，哪些是指定题目（select_pro）的作答 2. 将这些题目的作答改为1（对），模拟抄袭特定题目的答案
        select_idx = torch.isin(qa_array[:num], torch.tensor(select_pro)).tolist()
        qa_array[:num][select_idx] = 1
    elif bias_type == "guess":
        # 噪声类型：瞎猜（模拟学生随机蒙答案，0/1随机）
        # 1. 随机选择要注入噪声的位置 2. 对选中位置随机赋值0或1（各50%概率），模拟瞎猜
        select_idx = np.random.permutation(range(num))[0: int(num * bias_p)]
        qa_array[select_idx] = torch.tensor(np.random.choice([0, 1], size=len(select_idx), p=[0.5, 0.5]))
    return qa_array


def inject_noise(qa_dataArray ,interactions_per_student, student_p, bias_p, bias_type, datasetname, return_noise_label=False):
    # 对整个数据集的学生作答序列进行批量噪声注入：
    # 筛选出要注入噪声的学生（按作答次数排序，通常选作答多的学生，更贴近真实场景）；
    # 对选中的学生，调用select_bias注入指定类型的噪声；
    # 可选返回 “噪声标签”（标记哪些学生被注入了噪声），用于后续评估。
    idxes = np.argsort(interactions_per_student)[::-1]
    # 1. 按学生的有效作答次数降序排序，返回排序后的索引
    num_of_studnet = len(qa_dataArray)
    #  2. 学生总数
    perm = idxes[0: int(num_of_studnet * student_p)]
    #  3. 选择前 int(num_of_studnet * student_p) 个学生（作答多的学生）进行噪声注入
    cnt = 0
    #  记录被注入噪声的学生数
    # import ipdb; ipdb.set_trace()
    if bias_type == 'plag_by_pro':
        # 加载题目信息文件（存储题目ID和频次/难度等）
        pro_cnt_path = f'data/{datasetname}/question.json'
        with open(pro_cnt_path, 'r') as f:
            pro_cnt = json.load(f)
            # 提取题目ID并按指定规则排序（如按频次降序），选前 int(bias_p*总数) 个题目作为抄袭目标
        pro_cnt_descending = [int(point['id']) + 1 for point in pro_cnt]
        select_pro = pro_cnt_descending[: int(len(pro_cnt_descending) * bias_p)]
    else:
        select_pro = None
    for i in range(num_of_studnet):
        qa_array = qa_dataArray[i] # 第i个学生的作答序列
        num = interactions_per_student[i] # 第i个学生的有效作答次数
        # data = [q_dataArray[i], qa_dataArray[i], p_dataArray[i], interactions_per_student[i]]
        if i in perm:# 该学生被选中注入噪声
            # import ipdb; ipdb.set_trace()
            # print(qa_dataArray[i][:])
            # 调用select_bias注入噪声，修改作答序列
            qa_dataArray[i][:] = select_bias(qa_array, num, bias_type, bias_p, select_pro)
            # print(qa_dataArray[i][:])
            cnt += 1  # 计数+1
    print("Injecting Over: " + str(cnt) + "Bad Seqs, " + str(num_of_studnet - cnt) + "Clean Seqs")
    if not return_noise_label:
        return qa_dataArray
    else:
        # 生成噪声标签：1表示该学生被注入噪声，0表示干净
        noise_label = np.zeros(num_of_studnet)
        noise_label[perm] = 1
        return qa_dataArray, noise_label

class PID_DATA(object):
    """
    DACE PID_DATA
    适配 F19 / S19（CSedM）数据：
    - 使用 code_seq（代码文本）
    - 返回 code mask & code seq length
    """

    def __init__(
        self,
        n_question,
        seqlen,
        dataset_name,
        code_model_name="/home/cuiying/packages/codebert-base",
        device="cuda",
        separate_char=",",
        base_cache_dir="./code_cache"
    ):
        self.n_question = n_question
        self.seqlen = seqlen
        self.separate_char = separate_char
        self.dataset_name = dataset_name

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(code_model_name)
        self.model = AutoModel.from_pretrained(code_model_name).to(device)
        self.model.eval()

        self.cache_dir = os.path.join(base_cache_dir, dataset_name)
        os.makedirs(self.cache_dir, exist_ok=True)

    # --------------------------------------------------
    # 原始 DACE：CodeBERT 编码
    # --------------------------------------------------
    def encode_code(self, code_list, batch_size=16):
        embeddings = []
        for i in range(0, len(code_list), batch_size):
            batch = code_list[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**enc)
                cls_emb = out.last_hidden_state[:, 0, :]
                embeddings.append(cls_emb.cpu())

        return torch.cat(embeddings, dim=0)

    # --------------------------------------------------
    # 核心函数：load_data
    # --------------------------------------------------
    def load_data(self, jsonl_path, cache_name="train"):
        cache_file = os.path.join(self.cache_dir, f"{cache_name}_code.npy")

        q_data, qa_data, p_data, code_data = [], [], [], []
        code_len_data = []
        code_seq_len_data = []

        all_code_texts = []
        user_records = []

        # ==================================================
        # 1. 读取 jsonl
        # ==================================================
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)

                pid_seq = record["pid_seq"]
                qid_seq = record["qid_seq"]
                a_seq = record["a_seq"]
                code_seq = record["code_seq"]

                assert len(pid_seq) == len(qid_seq) == len(a_seq) == len(code_seq)

                user_records.append(
                    (pid_seq, qid_seq, a_seq, code_seq)
                )
                all_code_texts.extend(code_seq)

        # ==================================================
        # 2. CodeBERT 编码（cache 机制，完全不变）
        # ==================================================
        if os.path.exists(cache_file):
            all_code_emb = torch.from_numpy(np.load(cache_file))
        else:
            all_code_emb = self.encode_code(all_code_texts)
            np.save(cache_file, all_code_emb.numpy())

        # ==================================================
        # 3. 序列切分 + padding（原始 DACE）
        # ==================================================
        idx = 0
        for pid_seq, qid_seq, a_seq, code_seq in user_records:
            seq_len = len(code_seq)
            n_split = math.ceil(seq_len / self.seqlen)

            for k in range(n_split):
                start = k * self.seqlen
                end = min((k + 1) * self.seqlen, seq_len)
                cur_len = end - start

                qs, ps, qas, cs = [], [], [], []

                for i in range(start, end):
                    q = int(qid_seq[i])
                    p = int(pid_seq[i])
                    a = int(a_seq[i])
                    qa = q + a * self.n_question

                    qs.append(q)
                    ps.append(p)
                    qas.append(qa)
                    cs.append(all_code_emb[idx].numpy())
                    idx += 1

                pad_len = self.seqlen - cur_len
                if pad_len > 0:
                    qs.extend([0] * pad_len)
                    ps.extend([0] * pad_len)
                    qas.extend([0] * pad_len)
                    cs.extend([np.zeros(all_code_emb.shape[1])] * pad_len)

                q_data.append(qs)
                p_data.append(ps)
                qa_data.append(qas)
                code_data.append(cs)

                mask = [1] * cur_len + [0] * pad_len
                code_len_data.append(mask)
                code_seq_len_data.append(cur_len)# 新加的一行

        q_array = np.array(q_data, dtype=np.int64)
        p_array = np.array(p_data, dtype=np.int64)
        qa_array = np.array(qa_data, dtype=np.int64)
        code_array = np.array(code_data, dtype=np.float32)
        code_len_array = np.array(code_len_data, dtype=np.int64)  # >>> ADD
        code_seq_len_array = np.array(code_seq_len_data, dtype=np.int64)

        return q_array, qa_array, p_array, code_array, code_len_array, code_seq_len_array
