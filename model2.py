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
import optuna

class RankNet(nn.Module):
    def __init__(self, trial, num_layer, input_dim, h_dim, lr_rate):
        super(RankNet, self).__init__()
        self.activation = get_activation(trial)
        # 第1層
        self.fc = nn.ModuleList([nn.Linear(input_dim, h_dim[0])])
        # 第2層以降
        for i in range(1, num_layer):
            self.fc.append(nn.Linear(h_dim[i-1], h_dim[i]))
        self.dropout = nn.Dropout(lr_rate)

        self.fc_last = nn.Linear(h_dim[i], 1)

    def forward(self, x):
        for i, l in enumerate(self.fc):
            x = self.dropout(x)
            #nr_init(x.weight)
            x = l(x)
            x = self.activation(x)
        x = self.fc_last(x)
        return x

    def loss(self, torch_batch_rankings, torch_batch_std_labels):

        # Make a pair from the model predictions
        batch_pred = self.forward(torch_batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1) # batch_pred_dim = [40]
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

    def predict(self, x):
        return self.forward(x)

def train(model, train_data, optimizer):
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
        for k in [1, 5, 10]:
            if len(label_ar) > k:
                ndcg_s = ndcg_score_tensor(label_ar, y_pred_sorted, k=k)
                if not math.isnan(ndcg_s):
                    ndcg_ls[k].append(ndcg_s)

            else:
                denominator[k].append(len(label_ar))

    for k in [1, 5, 10]:
        # Subtraction for the number of documents less than k
        ndcg_k[k] = sum(ndcg_ls[k]) / (len(ndcg_ls[k]) - len(denominator[k]))

    return ndcg_k[5]

def get_optimizer(trial, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    return optimizer

def get_activation(trial):
    activation = F.relu
    return activation


def test2(model, test_data):
    # Testing
    ks=[1, 3, 5, 10]
    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for qid, batch_ranking, batch_labels in test_data: # _, [batch, ranking_size, num_features], [batch, ranking_size]

        if torch.sum(batch_labels) <=0: # filter dumb queries
          continue

        batch_rele_preds = model.predict(batch_ranking)
        pred_ar = batch_rele_preds.squeeze(1)
        # _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        # batch_sorted_inds = batch_sorted_inds.squeeze()
        # batch_labels = batch_labels.squeeze()
        # print(batch_sorted_inds.size())
        # print(batch_labels.size())
        # exit()
        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=batch_sys_sorted_labels, batch_ideal_sorted_labels=batch_ideal_sorted_labels, ks=ks)

        # default batch_size=1 due to testing data
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.squeeze(batch_ndcg_at_ks, dim=0))
        cnt += 1

    avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
    print(avg_ndcg_at_ks)

def objective(trial):
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
    num_layer = trial.suggest_int('num_layer', 3, 7)

    h_dim = [int(trial.suggest_discrete_uniform("h_dim_"+str(i), 16, 128, 16)) for i in range(num_layer)]

    lr_rate = trial.suggest_uniform("dropout_l", 0.2, 0.5)

    ranknet = RankNet(trial, num_layer, 46, h_dim, lr_rate)
    optimizer = get_optimizer(trial, ranknet)
    for step in range(EPOCH):
      train(ranknet, train_data, optimizer)
      ndcg = test2(ranknet, test_data)

    print('ndcg=',ndcg)
    return ndcg




if __name__ == '__main__':
  EPOCH = 2
  k_fold = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
  torch.manual_seed(1)

  data_meta = get_data_meta(data_id='MQ2007_Super')

  for fold in k_fold:
    TRIAL_SIZE = 1
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=TRIAL_SIZE)
    print(study.best_params)
    print(study.best_value)
