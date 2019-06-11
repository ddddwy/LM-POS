from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config
import data
from numpy import random
import pandas as pd

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class CFSM(nn.Module):
    def __init__(self):
        super(CFSM, self).__init__()
        # cluster initailization
        self.vocab2cluster, self.cluster2vocab, self.cluster, self.word_num = self.gen_cluster(config.vocab_path, 
                                                                                               config.cluster_path)
        self.cluster_num = len(self.cluster.keys())
        self.mask_neg, self.mask_pos = self.gen_cluster_mask()
        
        # computation layers
        self.psi = nn.Linear(config.hidden_dim, self.cluster_num, bias=False)
        self.phi_list = []
        for i in range(self.cluster_num):
            phi = nn.Linear(config.hidden_dim, self.word_num, bias=True)
            self.phi_list.append(phi)
        self.phi_list = nn.ModuleList(self.phi_list)
        
        
    def forward(self, h_p, target_cluster):
        p_c = F.softmax(self.psi(h_p), dim=1) # B x class_num
        
        if target_cluster is None:
            target_cluster = torch.argmax(p_c, dim=1)
        
        p_w = torch.zeros((config.batch_size, self.word_num)) # B x word_num
        if use_cuda:
            p_w = p_w.cuda()
            
        for b_idx in range(config.batch_size):
            # find the corresponding cluster according to ground truth cid
            cid = target_cluster[b_idx].item()
            phi_c = self.phi_list[cid]
            # compute the word dist under current cluster
            phi_out = phi_c(h_p[b_idx])
            # if there exists negative values in the masking part, we set it to be positive
            filter_out = torch.where(phi_out>0, phi_out, phi_out * self.mask_pos[cid])
            # we multiply the masking part with -1e5, to set its output to be zero
            filter_out = filter_out * self.mask_neg[cid]
            out = F.softmax(filter_out, dim=0)
            p_w[b_idx] = out
        
        return p_c, p_w
        
    
    def gen_cluster(self, vocab_path, cluster_path):
        '''
        Extract the brown clusters, build three mappings:
            vocab2cluster: wid -> (cid, cwid)
            cluster2vocab: (cid, cwid) -> wid
            cluster: cid -> [wid, wid, ...]
        '''
        vocab = pd.read_csv(vocab_path, sep=' ', header=None)
        vocab_list = vocab.loc[:, 0].tolist()
        vocab_new_list = vocab_list
        vocab_new_list.insert(0, '[STOP]')
        vocab_new_list.insert(0, '[START]')
        brown_cluster = pd.read_csv(cluster_path, sep='\t', header=None, names=['cid', 'word', 'freq'], dtype=str)
        unique_cids = brown_cluster.cid.unique()
        
        vocab2cluster = {}
        cluster2vocab = {}
        cluster = {}
        cluster_out = {}
        # [PAD], [UNK], [START] and [STOP] get the cwids 0,1,2,3.
        vocab2cluster[0] = (0, 0)
        cluster2vocab[(0, 0)] = 0
        vocab2cluster[1] = (0, 1)
        cluster2vocab[(0, 1)] = 1
        cluster[0] = [0, 1] 
        cluster_out[0] = ['[PAD]', '[UNK]']
        for cid, cid_str in enumerate(unique_cids):
            current_cluster = brown_cluster[brown_cluster.cid == cid_str]
            for i in range(len(current_cluster)):
                word = current_cluster.iloc[i,1]
                if word not in vocab_new_list:
                    continue
                wid = vocab_new_list.index(word)
                ccid = cid + 1
                if ccid not in cluster.keys():
                    wid += 2
                    vocab2cluster[wid] = (ccid, 0)
                    cluster2vocab[(ccid, 0)] = wid
                    cluster[ccid] = [wid]
                    cluster_out[ccid] = [word]
                else:
                    wid += 2
                    cwid = len(cluster[ccid])
                    vocab2cluster[wid] = (ccid, cwid)
                    cluster2vocab[(ccid, cwid)] = wid
                    cluster[ccid].append(wid)
                    cluster_out[ccid].append(word)
                    
        word_num = 0
        for value in cluster.values():
            if word_num < len(value):
                word_num = len(value)
                
        return vocab2cluster, cluster2vocab, cluster, word_num
    
    
    def gen_cluster_mask(self):
        '''
        Generate the masking matrix for each cluster
        '''
        # initialize the mask with ones
        cluster_mask_neg = torch.ones((self.cluster_num, self.word_num))
        cluster_mask_pos = torch.ones((self.cluster_num, self.word_num))
        if use_cuda:
            cluster_mask_neg = cluster_mask_neg.cuda()
            cluster_mask_pos = cluster_mask_pos.cuda()
            
        for i in range(self.cluster_num):
            c_size = len(self.cluster[i])
            for j in range(self.word_num):
                if j >= c_size:
                    cluster_mask_neg[i, j] = -1e5
                    cluster_mask_pos[i, j] = -1
        
        return cluster_mask_neg, cluster_mask_pos
        

    def batch_vocab2cluster(self, target, vocab):
        '''
        Given a batch of ground truth words,
        Return its corresponding cid (target_cluster), cwid (target_word).
        '''
        words_truth = []
        cluster_truth = []
        for t_id in range(target.size()[0]):
            wid = target[t_id].item()
            if wid in self.vocab2cluster.keys():
                (cid, cwid) = self.vocab2cluster[wid]
                cluster_truth.append(cid)
                words_truth.append(cwid)
            else:
                wid = vocab.word2id(data.UNKNOWN_TOKEN)
                (cid, cwid) = self.vocab2cluster[wid]
                cluster_truth.append(cid)
                words_truth.append(cwid)
            
        target_word = Variable(torch.LongTensor(words_truth))
        target_cluster = Variable(torch.LongTensor(cluster_truth))
        if use_cuda:
            target_word = target_word.cuda()
            target_cluster = target_cluster.cuda()
            
        return target_cluster, target_word
    
    
    def batch_decode(self, cid_batch, cwid_batch, vocab):
        '''
        Given two batches of predicted cid and cwid,
        Convert it back to wid
        '''
        latest_tokens = []
        for b_idx in range(config.batch_size):
            cid = cid_batch[b_idx].item()
            cwid = cwid_batch[b_idx].item()
            if (cid, cwid) in self.cluster2vocab.keys():
                next_wid = self.cluster2vocab[(cid, cwid)]
            else:
                next_wid = vocab.word2id(data.UNKNOWN_TOKEN)
            latest_tokens.append(next_wid)
        latest_batch = Variable(torch.LongTensor(latest_tokens))
        if use_cuda:
            latest_batch = latest_batch.cuda()
            
        return latest_batch   