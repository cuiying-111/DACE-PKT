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
    # "CodeNet": {
    #     'n_pid': ...,
    #     'n_question': ...,
    #     'seqlen': ...,
    #     'data_dir': '...',
    #     'data_name': 'CodeNet'
    # }
}


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ========= dataset =========
    dataset = "BePKT"

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
