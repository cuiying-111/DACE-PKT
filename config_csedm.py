# ================= Dataset Statistics =================
import torch

data_statics = {
    "BePKT": {
        'n_pid': 607,
        'n_question': 107,
        'seqlen': 200,
        'data_dir': '/home/cuiying/projects_paper/DACE-main/data/bepkt/bepkt_with_code_seq',
        'data_name': 'BePKT_pid'
    },
    "AtCoder": {
        # 题目数（不同 pid 的数量）
        # 👉 来自 step5 / step6 之后的 pid 映射
        'n_pid': 1518,  # 【先占位，下面我告诉你怎么精确算】
        # qid 的总数（注意：不是 interaction 数）
        'n_question': 1518,  # 【先占位】
        # DACE 原始 seqlen，建议先不改
        'seqlen': 200,
        # 你 step5 生成的 jsonl 所在目录
        'data_dir': '/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/AtCoder_with_code_seq',
        # 这个名字只用于 logging / cache / save
        'data_name': 'AtCoder_pid'
    },
    "AIZU": {
        'n_pid': 2525,  # 【先占位】
        'n_question': 2525,  # 【先占位】
        'seqlen': 200,
        'data_dir': '/home/cuiying/projects_paper/DACE-main/data/codenet/Project_CodeNet/AIZU_with_code_seq',
        'data_name': 'AIZU_pid'
    },
    "csedm_f": {
        'n_pid':  50,        # pid 数
        'n_question': 50,    # 关键点：qid = pid
        'seqlen': 200,
        'data_dir': '/home/cuiying/projects_paper/DACE-main/data/csedm_f/f19_with_code_seq',
        'data_name': 'csedm_f_pid'
    },
    "csedm_s": {
        'n_pid':  50,
        'n_question': 50,
        'seqlen': 200,
        'data_dir': '/home/cuiying/projects_paper/DACE-main/data/csedm_s/s19_with_code_seq',
        'data_name': 'csedm_s_pid'
    }
}


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========= dataset =========
    dataset = "csedm_f"

    # ========= bias / noise =========
    bias_type = "None"
    bias_p = 0.0
    inject_p = 0.0

    # ========= training schedule =========
    fb_epoch = 10
    max_iter = 300

    # ========= model =========
    disentangle = 1
    model = "DACE_pid"
    model_type = "DACE"
    save = dataset

    # ========= dataset-dependent =========
    n_question = data_statics[dataset]['n_question']
    n_pid = data_statics[dataset]['n_pid']
    seqlen = data_statics[dataset]['seqlen']
    data_dir = data_statics[dataset]['data_dir']
    data_name = data_statics[dataset]['data_name']

    # ========= training =========
    train_set = 1
    seed = 224
    optim = 'adam'
    batch_size = 64
    lr = 1e-5
    maxgradnorm = -1
    final_fc_dim = 512
    l2 = 1e-5

    # ========= Knowledge State Extractor =========
    d_model = 256
    d_ff = 1024
    dropout = 0.1
    n_block = 1
    n_head = 8
    kq_same = 1

    # ========= logging =========
    file_name = (
        f"{dataset}_{model}_bias_{bias_type}_p_{bias_p}_seed_{seed}"
    )
