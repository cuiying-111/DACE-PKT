# Code reused from https://github.com/jennyzhang0215/DKVMN.git

import numpy as np
import torch
import math
from sklearn import metrics
from utils import model_isPid_type
from tqdm import tqdm

transpose_data_model = {'DACE'}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

def get_sequence_z(z, mask):
    '''
    z : (bs, seq_len, embed_dim)
    mask: (bs, seq_len)
    '''
    # import ipdb; ipdb.set_trace()

    mask = mask.unsqueeze(-1)
    z_hidden = (z * mask).sum(1) / mask.sum()
    # assert z_hidden.shape[1] == mask.shape[1]
    return z_hidden

def train(net, params, optimizer,
          q_data, qa_data, pid_data,
          code_emb_data, code_seq_len,
          label):

    net.train()
    pid_flag, model_type = model_isPid_type(params.model)

    N = int(math.ceil(len(q_data) / params.batch_size))

    # ===============================
    # Transpose to (seq_len, num_seq)
    # ===============================
    q_data = q_data.T
    qa_data = qa_data.T
    # code_emb_data is 3D: (N, T, D) -> (T, N, D)
    code_emb_data = np.transpose(code_emb_data, (1, 0, 2))


    # ===============================
    # Shuffle by student dimension
    # ===============================
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)

    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    code_emb_data = code_emb_data[:, shuffled_ind]
    code_seq_len = code_seq_len[shuffled_ind]

    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    true_el = 0

    for idx in range(N):
        optimizer.zero_grad()

        bs = params.batch_size
        sl = slice(idx * bs, (idx + 1) * bs)

        # ===============================
        # Slice batch
        # ===============================
        q_one_seq = q_data[:, sl]
        qa_one_seq = qa_data[:, sl]
        code_emb_one = code_emb_data[:, sl]
        code_len_one = code_seq_len[sl]    # (bs,)

        if pid_flag:
            pid_one_seq = pid_data[:, sl]

        # ===============================
        # Transpose for DACE
        # ===============================
        if model_type in transpose_data_model:
            input_q = q_one_seq.T
            input_qa = qa_one_seq.T
            target = qa_one_seq.T
            # code_emb_one is 3D: (T, batch_size, D) -> (batch_size, T, D)
            code_emb = np.transpose(code_emb_one, (1, 0, 2))
            code_len = code_len_one   # (bs,)
            if pid_flag:
                input_pid = pid_one_seq.T
        else:
            input_q = q_one_seq
            input_qa = qa_one_seq
            target = qa_one_seq
            code_emb = code_emb_one
            code_len = code_len_one
            if pid_flag:
                input_pid = pid_one_seq

        # ===============================
        # Target construction (unchanged)
        # ===============================
        target = (target - 1) / params.n_question
        target_1 = np.floor(target)

        # ===============================
        # To torch
        # ===============================
        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target_t = torch.from_numpy(target_1).float().to(device)
        code_emb = torch.from_numpy(code_emb).float().to(device)
        code_len = torch.from_numpy(code_len).long().to(device)

        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)
            loss, pred, true_ct = net(
                input_q, input_qa, target_t, input_pid,
                code_emb=code_emb, code_len=code_len
            )
        else:
            loss, pred, true_ct = net(
                input_q, input_qa, target_t,
                code_emb=code_emb, code_len=code_len
            )

        # ===============================
        # Backprop
        # ===============================
        loss.backward()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm
            )

        optimizer.step()

        pred = pred.detach().cpu().numpy()
        true_el += true_ct.cpu().numpy()

        # ===============================
        # Collect non-padding
        # ===============================
        target_flat = target_1.reshape(-1)
        nopadding_index = np.flatnonzero(target_flat >= -0.9)

        pred_list.append(pred[nopadding_index])
        target_list.append(target_flat[nopadding_index])

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


# 这是fc主要用的训练的函数
def train_clean(params,
                model_fb, model_fc,
                optim_fc,
                disc, optim_disc,
                q_data, qa_data, pid_data,
                code_emb_data, code_seq_len,
                label, epoch):

    model_fb.eval()
    model_fc.train()

    pid_flag, model_type = model_isPid_type(params.model)
    N = int(math.ceil(len(q_data) / params.batch_size))

    # ===============================
    # Transpose
    # ===============================
    q_data = q_data.T
    qa_data = qa_data.T
    # code_emb_data is 3D: (N, T, D) -> (T, N, D)
    code_emb_data = np.transpose(code_emb_data, (1, 0, 2))


    # ===============================
    # Shuffle
    # ===============================
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)

    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]
    code_emb_data = code_emb_data[:, shuffled_ind]
    code_seq_len = code_seq_len[shuffled_ind]

    if pid_flag:
        pid_data = pid_data.T
        pid_data = pid_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    true_el = 0

    # ===============================
    # preprocess helper
    # ===============================
    def preprocess(idx):
        bs = params.batch_size
        sl = slice(idx * bs, (idx + 1) * bs)

        q_one_seq = q_data[:, sl]
        qa_one_seq = qa_data[:, sl]
        code_emb_one = code_emb_data[:, sl]
        code_len_one = code_seq_len[sl]   # (bs,)

        if pid_flag:
            pid_one_seq = pid_data[:, sl]

        if model_type in transpose_data_model:
            input_q = q_one_seq.T
            input_qa = qa_one_seq.T
            target = qa_one_seq.T
            # code_emb_one is 3D: (T, batch_size, D) -> (batch_size, T, D)
            code_emb = np.transpose(code_emb_one, (1, 0, 2))
            code_len = code_len_one
            if pid_flag:
                input_pid = pid_one_seq.T
        else:
            input_q = q_one_seq
            input_qa = qa_one_seq
            target = qa_one_seq
            code_emb = code_emb_one
            code_len = code_len_one
            if pid_flag:
                input_pid = pid_one_seq

        target = (target - 1) / params.n_question
        target_1 = np.floor(target)

        input_q = torch.from_numpy(input_q).long().to(device)
        input_qa = torch.from_numpy(input_qa).long().to(device)
        target_t = torch.from_numpy(target_1).float().to(device)
        code_emb = torch.from_numpy(code_emb).float().to(device)
        code_len = torch.from_numpy(code_len).long().to(device)

        if pid_flag:
            input_pid = torch.from_numpy(input_pid).long().to(device)
        else:
            input_pid = None

        return input_q, input_qa, target_t, input_pid, code_emb, code_len, target_1

    # ======================================================
    # 1️⃣ Train discriminator
    # ======================================================
    if params.disentangle:
        for idx in range(N):
            input_q, input_qa, target, input_pid, code_emb, code_len, _ = preprocess(idx)

            # fb & fc receive IDENTICAL inputs
            loss_c, _, _, z_c = model_fc(
                input_q, input_qa, target, input_pid,
                code_emb=code_emb, code_len=code_len,
                return_output=True
            )

            with torch.no_grad():
                loss_b, _, _, z_b = model_fb(
                    input_q, input_qa, target, input_pid,
                    code_emb=code_emb, code_len=code_len,
                    return_output=True
                )

            z_hidden_b = get_sequence_z(z_b, target > -0.9).detach()
            z_hidden_c = get_sequence_z(z_c, target > -0.9).detach()

            dis_loss = -disc(z_hidden_b, z_hidden_c)

            optim_disc.zero_grad()
            dis_loss.backward()
            optim_disc.step()
            disc.spectral_norm()

    # ======================================================
    # 2️⃣ Train clean extractor
    # ======================================================
    for idx in range(N):
        input_q, input_qa, target, input_pid, code_emb, code_len, target_1 = preprocess(idx)

        loss_c, pred, true_ct, z_c = model_fc(
            input_q, input_qa, target, input_pid,
            code_emb=code_emb, code_len=code_len,
            return_output=True
        )

        with torch.no_grad():
            loss_b, _, _, z_b = model_fb(
                input_q, input_qa, target, input_pid,
                code_emb=code_emb, code_len=code_len,
                return_output=True
            )

        z_hidden_b = get_sequence_z(z_b, target > -0.9).detach()
        z_hidden_c = get_sequence_z(z_c, target > -0.9)

        dis_loss = disc(z_hidden_b, z_hidden_c)

        loss = loss_c
        if params.disentangle:
            loss += dis_loss

        optim_fc.zero_grad()
        loss.backward()
        optim_fc.step()

        pred = pred.detach().cpu().numpy()
        true_el += true_ct.cpu().numpy()

        target_flat = target_1.reshape(-1)
        nopadding_index = np.flatnonzero(target_flat >= -0.9)

        pred_list.append(pred[nopadding_index])
        target_list.append(target_flat[nopadding_index])

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc


def test(net, params,
         q_data, qa_data, pid_data,
         code_emb_data, code_len_data,
         label='Test'):

    net.eval()
    total_loss = 0.0

    all_pred, all_target = [], []

    pid_flag, model_type = model_isPid_type(params.model)

    with torch.no_grad():
        for idx in tqdm(range(q_data.shape[0]), desc=label):

            # ===============================
            # prepare input
            # ===============================
            q = torch.LongTensor(q_data[idx]).unsqueeze(0).to(device)
            qa = torch.LongTensor(qa_data[idx]).unsqueeze(0).to(device)

            if pid_flag:
                pid = torch.LongTensor(pid_data[idx]).unsqueeze(0).to(device)
            else:
                pid = None

            code_emb = torch.FloatTensor(code_emb_data[idx]).unsqueeze(0).to(device)
            code_len = torch.LongTensor([code_len_data[idx]]).to(device)

            # ===============================
            # target & mask (same as train_clean)
            # ===============================
            target = (qa - 1) / params.n_question
            target = torch.floor(target)

            # ===============================
            # forward (same interface as train_clean)
            # ===============================
            loss_c, pred, _, _ = net(
                q, qa, target, pid,
                code_emb=code_emb,
                code_len=code_len,
                return_output=True
            )

            total_loss += loss_c.item()

            # ===============================
            # flatten valid positions
            # ===============================
            pred = pred.detach().cpu().numpy()
            target_np = target.cpu().numpy()

            target_flat = target_np.reshape(-1)
            pred_flat = pred.reshape(-1)

            nopadding_index = np.flatnonzero(target_flat >= -0.9)

            all_pred.append(pred_flat[nopadding_index])
            all_target.append(target_flat[nopadding_index])

    all_pred = np.concatenate(all_pred)
    all_target = np.concatenate(all_target)

    auc = compute_auc(all_target, all_pred)
    acc = compute_accuracy(all_target, all_pred)

    return total_loss / q_data.shape[0], acc, auc
