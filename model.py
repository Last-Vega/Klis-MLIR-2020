import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ptranking.data.data_utils import LTRDataset, SPLIT_TYPE
import torch.nn as nn
from ptranking.base.ranker import NeuralRanker
from ptranking.ltr_adhoc.eval.parameter import ScoringFunctionParameter
from ptranking.data.data_utils import get_data_meta
from ptranking.ltr_adhoc.eval.eval_utils import ndcg_at_ks, ndcg_at_k
from ptranking.metric.adhoc_metric import torch_nDCG_at_k, torch_nDCG_at_ks
import torch.nn as nn
import torch.nn.functional as F
from ptranking.base.ranker import NeuralRanker
from collections import defaultdict

class MLIRSFP(ScoringFunctionParameter):
    def __init__(self):
        super(MLIRSFP, self).__init__()

    def default_para_dict(self):
        ffnns_para_dict = dict(num_layers=5, HD_AF='R', HN_AF='R', TL_AF='S', apply_tl_af=True, BN=True, RD=False, FBN=False)

        sf_para_dict = dict()
        sf_para_dict['id'] = self.model_id
        sf_para_dict[self.model_id] = ffnns_para_dict

        self.sf_para_dict = sf_para_dict
        return sf_para_dict

input_dim = 46
D_in = 128
H1 = 64
H2 = 32
layers = [
            nn.Linear(input_dim, D_in),
            nn.ReLU(),
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, 1)
          ]
# print(layers[-1])
# exit()
class RankNet(nn.Module):
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers)
        self.fc = layers
        self.dropout = nn.Dropout(0.01)
        self.activation = F.relu
        self.fc_last = layers[-1]

    def forward(self, x):
        for i, l in enumerate(self.fc):
            x = self.dropout(x)
            # print('dropout',x)
            #nr_init(x.weight)
            x = l(x)
            x = self.activation(x)
            # print('activation',x)
        # x = self.fc_last(x)
        return x

    def loss(self, torch_batch_rankings, torch_batch_std_labels):

        # Make a pair from the model predictions
        batch_pred = self.forward(torch_batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1) # batch_pred_dim = [40]
        # print('batch_pred_dim', batch_pred_dim)
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]

        # Make a pair from the relevance of the label
        batch_std = torch_batch_std_labels # batch_std = [40]
        batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]

        # Align to -1 ~ 1
        batch_Sij = torch.clamp(batch_std_diffs, -1, 1)

        sigma = 1.0
        batch_loss_1st = 0.5 * sigma * batch_pred_diffs * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_diffs) + 1.0)

        # Calculate loss outside diagonal
        diagona = 1 - torch.eye(batch_loss_1st.shape[0])
        batch_loss = (batch_loss_1st + batch_loss_2nd) * diagona
        combination = (batch_loss_1st.shape[0] * (batch_loss_1st.shape[0] - 1)) / 2

        batch_loss_triu = (torch.sum(batch_loss) / 2) / combination

        #print(batch_loss_triu)

        return batch_loss_triu

    # def predict(self, x):
    #     return self.forward(x)

    # def forward(self, batch_ranking=None, batch_stds_labels=None, sigma=1.0):
    #     s_batch = self.model(batch_ranking) #スコア計算
    #     pred_diff = s_batch - s_batch.view(1, -1) #s_i - s_j 行列
    #     #対角成分削除
    #     row_inds, col_inds = np.triu_indices(batch_ranking.size()[0], k=1)
    #     si_sj = pred_diff[row_inds, col_inds] #上三角s_i - s_j 行列


def train(model, optimizer, train_data):
    model.train()
    for qid, torch_batch_rankings, torch_batch_std_labels in train_data:
        data, target = torch_batch_rankings, torch_batch_std_labels
        optimizer.zero_grad()
        loss = model.loss(data, target)
        loss.backward()
        optimizer.step()


def test(model, test_data):
    model.eval()
    ndcg_ls = defaultdict(list)
    ndcg_k = {}
    denominator = defaultdict(list)
    for qid, data, target in test_data:
        pred = model.predict(data)
        pred_ar = pred.squeeze(1)
        label_ar = target
        _, order = torch.sort(pred_ar, descending=True)
        y_pred_sorted = label_ar[order]
        for k in [1, 3, 5, 10]:
            if len(label_ar) > k:
                ndcg_s = ndcg_score_tensor(label_ar, y_pred_sorted, k=k)
                if not math.isnan(ndcg_s):
                    ndcg_ls[k].append(ndcg_s)

            else:
                denominator[k].append(len(label_ar))

    for k in [1, 3, 5, 10]:
        # Subtraction for the number of documents less than k
        ndcg_k[k] = sum(ndcg_ls[k]) / (len(ndcg_ls[k]) - len(denominator[k]))

    return ndcg_k[5]


train_file = './vali_as_train.txt'
test_file = './test.txt'

train_data = LTRDataset(
                        data_id='MQ2007_Super',
                        file=train_file,
                        split_type=SPLIT_TYPE.Train,
                        batch_size=1,
                        shuffle=True,
                        presort=True,
                        data_dict=None,
                        eval_dict=None,
                        buffer=False
                    )

test_data = LTRDataset(
                        data_id='MQ2007_Super',
                        file=test_file,
                        split_type=SPLIT_TYPE.Test,
                        shuffle=False,
                        data_dict=None,
                        batch_size=1,
                        buffer=False
                    )

data_meta = get_data_meta(data_id='MQ2007_Super')

# sf_para_dict = MLIRSFP().default_para_dict()
#
# sf_para_dict['ffnns'].update(dict(num_features=data_meta['num_features']))
ranknet = RankNet(layers=layers)
# ranknet = RankNet(trial, num_layer, 46, h_dim, lr_rate)
optimizer = torch.optim.Adam(ranknet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


batch_loss = train(ranknet, optimizer, train_data)
print(batch_loss)

# ks=[1, 5, 10]
# sum_ndcg_at_ks = torch.zeros(len(ks))
# cnt = torch.zeros(1)
# already_sorted = True if test_data.presort else False
# for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]
#
#     if torch.sum(batch_labels) <=0: # filter dumb queries
#       continue
#
#     batch_rele_preds = ranknet.predict(batch_ranking)
#
#     _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)
#
#     batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
#     if already_sorted:
#         batch_ideal_sorted_labels = batch_labels
#     else:
#         batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)
#
#     batch_ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=batch_sys_sorted_labels, batch_ideal_sorted_labels=batch_ideal_sorted_labels, ks=ks)
#
#     # default batch_size=1 due to testing data
#     sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.squeeze(batch_ndcg_at_ks, dim=0))
#     cnt += 1
#
# avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
# print(avg_ndcg_at_ks)
