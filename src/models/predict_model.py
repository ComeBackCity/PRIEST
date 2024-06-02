from statistics import mode
import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from src.utils import utils, validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, \
    roc_auc_score, average_precision_score
from torchmetrics.functional import auroc, average_precision
from scipy import interp
import os
import time


# def extract_selfattention_maps(model, model_type, x_test ,mask,src_key_padding_mask):
#     model = load_model(
#         model=model,
#         path=f'checkpoint/6/{model_type}/best_model/checkpoint.pth'
#     )

#     transformer_encoder = model

    

#     seq_len = 9
#     batch_size = x_test.shape[1]

#     src_mask = torch.zeros((seq_len,seq_len)).bool()
#     src_key_padding_mask = torch.zeros((batch_size,seq_len)).bool()

#     attention_maps = extract_self_attn_maps(transformer_encoder,x,src_mask,src_key_padding_mask)


def extract_self_attn_maps(transformer_encoder,x,mask,src_key_padding_mask):
    attention_maps = []
    num_layers = transformer_encoder.num_layers
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            attention_maps.append(attn)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attention_maps


def load_model(model, path):
    print(path)
    if not os.path.exists(path=path):
        print('Path does not exist')
        return

    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])

    return model


def test_model(model, model_type, x_test, y_test, device):
    seq_length = x_test.shape[0]
    model = load_model(
        model=model,
        # path=f'checkpoint/{seq_length}/{model_type}/best_model/checkpoint.pth'
        path=f'checkpoint/2023/{model_type}/best_model/checkpoint.pth'
    )

    if model_type == 'transformer' or model_type == 'transformer2' or model_type == 'transformer3' \
        or model_type == 'cnn' or model_type == 'cnn2' \
        or model_type == 'lstm1' or model_type == 'lstm2' or model_type == 'lstm3' or model_type == 'lstm4' \
        or model_type == 'mcma'  or model_type == 'mcma2' or model_type == 'mcma3' or model_type == 'mcma5' \
        or model_type == 'PRIEST':
        x_test = x_test.permute((1, 0, 2))

    torch.cuda.empty_cache()

    x_test = x_test.to(device)
    # time.sleep(10)
    y_test = y_test.to(device)
    # time.sleep(10)
    model.to(device)
    # print('Loaded model to GPU')
    # time.sleep(10)
    save_path = 'results_cov/attn/new_model/weights.pth'
    label_save_path = 'results_cov/labels.tensor'

    # save_path = None

    with torch.no_grad():
        model.eval()
        if model_type == 'transformer' or model_type == 'transformer2' or model_type == 'transformer3' \
            or model_type == 'cnn' or model_type == 'cnn2' \
            or model_type == 'lstm1' or model_type == 'lstm2' or model_type == 'lstm3' or model_type == 'lstm4'\
            or model_type == 'mcma'  or model_type == 'mcma2' or model_type == 'mcma3' or model_type == 'mcma5'\
            or model_type == 'PRIEST':
            logits, _ = model(x_test, debug=False)
        else:
            logits, _ = model(x_test, model.init_hidden(y_test.shape[0]))
        prob = F.softmax(logits, dim=1)
        labels = torch.argmax(prob, dim=1)
        torch.save(labels, label_save_path)
        pr_curve = precision_recall_curve(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
        roc_curv = roc_curve(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
        precision, recall, fscore, mcc, accuracy = validation.evaluate(y_test, labels)
        true_labels = F.one_hot(y_test, 2)
        auc = auroc(logits, true_labels, task='binary')
        ap = average_precision(logits, true_labels, task='binary')
        

        # [178  57 172  56  97]
        # seq_wise_acc = []
        # if model_type == 'our model':
        #     for i in range(0, x_test.shape[0], 149):
        #         seq_true = y_test[i:i+149].cpu().detach().numpy()
        #         seq_pred = labels[i:i+149].cpu().detach().numpy()
        #         seq_acc = accuracy_score(seq_true, seq_pred)
        #         seq_wise_acc.append(seq_acc)

        #     top_5_ind = np.argpartition(seq_wise_acc, -5)[-5:]
        #     top_5_val = np.asarray(seq_wise_acc)[top_5_ind]
        #     print(top_5_ind, top_5_val)



    sam = []
    if model_type == 'transformer' or model_type == 'transformer':
        sam = model.extract_self_attn_maps(src = x_test, device=device)
    
    return labels.detach().cpu().numpy(), precision, recall, fscore, mcc, accuracy, pr_curve, roc_curv, sam, auc, ap


# def test_model(model, model_type, x_test, y_test, device):
#     seq_length = x_test.shape[0]
#     model = load_model(
#         model=model,
#         path=f'checkpoint/{seq_length}/{model_type}/best_model/checkpoint.pth'
#     )

#     if model_type == 'transformer' or model_type == 'transformer2' or model_type == 'transformer3' \
#          or model_type == 'cnn' or model_type == 'cnn2' \
#          or model_type == 'lstm1' or model_type == 'lstm2' or model_type == 'lstm3' \
#         or model_type == 'our model':
#         x_test = x_test.permute((1, 0, 2))

#     # x_test = x_test.to(device)
    
#     model.to(device)
#     label_save_path = f'results_cov/{model_type}_labels.tensor'

#     with torch.no_grad():
#         model.eval()
#         logits_list = []
#         count = x_test.shape[0]
#         batch_size = 256
#         for i in range(0, count, batch_size):
#             batch = x_test[i:i+batch_size]
#             batch = batch.to(device)
#             if model_type == 'transformer' or model_type == 'transformer2' or model_type == 'transformer3' \
#                 or model_type == 'cnn' or model_type == 'cnn2' or model_type == 'our model' \
#                 or model_type == 'lstm1' or model_type == 'lstm2' or model_type == 'lstm3' or model_type == 'lstm4':
#                 logits, _ = model(batch, debug=True)
#             else:
#                 logits, _ = model(batch, model.init_hidden(y_test.shape[0]))

#             logits_list.append(logits)
#             torch.cuda.empty_cache()

#         logits = torch.cat(logits_list, dim=0)
        
#         y_test = y_test.to(device)
#         prob = F.softmax(logits, dim=1)
#         labels = torch.argmax(prob, dim=1)
#         torch.save(labels, label_save_path)
#         pr_curve = precision_recall_curve(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
#         roc_curv = roc_curve(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
#         auroc = roc_auc_score(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
#         auprc = average_precision_score(y_test.cpu().numpy(), prob[:, 1].cpu().numpy())
#         precision, recall, fscore, mcc, accuracy = validation.evaluate(y_test, labels)

#     sam = []
#     if model_type == 'transformer' or model_type == 'transformer':
#         sam = model.extract_self_attn_maps(src = x_test, device=device)
    
#     return labels.detach().cpu().numpy(), precision, recall, fscore, mcc, accuracy, pr_curve, roc_curv, sam, auroc, auprc

    