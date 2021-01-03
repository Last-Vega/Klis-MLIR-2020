import numpy as np
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
    def __init__(self, layers):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(*layers)
        # self.fc = layers
        # self.dropout = nn.Dropout(0.01)
        # self.activation = F.relu
        self.sigma = 1.0

    def forward(self, batch_ranking, batch_label):
        print('batch_ranking, batch_label',batch_ranking.size(), batch_label.size())
        batch_ranking = batch_ranking.reshape(0, 39)
        batch_s_ij = torch.unsqueeze(batch_ranking, dim=2) - torch.unsqueeze(batch_ranking, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j
        return batch_s_ij

    def cost(self, batch_ranking, batch_label):
        batch_s_ij = self.forward(batch_ranking, batch_label)

        batch_p_ij = 1.0 / (torch.exp(-self.sigma * batch_s_ij) + 1.0)

        batch_std_diffs = torch.unsqueeze(batch_label, dim=2) - torch.unsqueeze(batch_label, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

        print('batch_s_ij', batch_s_ij.size())
        print('batch_std_diffs', batch_std_diffs.size())
        print('batch_p_ij', batch_p_ij.size())
        print('batch_std_p_ij',batch_std_p_ij.size())

        # batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='mean')
        #
        # self.optimizer.zero_grad()
        # batch_loss.backward()
        # self.optimizer.step()

        # return batch_loss

def train(model, train_data, optimizer):
    model.train()
    for qid, torch_batch_rankings, torch_batch_std_labels in train_data:
        # print('batch_rankings, batch_std', torch_batch_rankings.size(), torch_batch_std_labels.size())
        data, target = torch_batch_rankings, torch_batch_std_labels
        # optimizer.zero_grad()
        loss = model.cost(data, target)
        break
        # loss.backward()
        # optimizer.step()



def get_optimizer(trial, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    return optimizer


def objective():
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
    ranknet = RankNet(layers)
    optimizer = get_optimizer(trial, ranknet)
    train(ranknet, train_data, optimizer)


trial = 2
objective()
