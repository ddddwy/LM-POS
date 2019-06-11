from __future__ import unicode_literals, print_function, division

import time
import numpy as np
import torch

import config
from batcher import Batcher
from data import Vocab
from train_util import get_input_from_batch, get_output_from_batch
from model import Model
from cfsm import CFSM

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)

        time.sleep(5)
        
        self.cfsm = CFSM()
        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, c_t_1 = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = get_output_from_batch(batch, use_cuda)
            
        with torch.no_grad():
            encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
            s_t_1 = self.model.reduce_state(encoder_hidden)
    
            step_losses = []
            for di in range(min(max_dec_len, config.max_dec_steps)):
                # use ground truth cluster label
                y_t_1 = dec_batch[:, di]  
                
                # use ground truth word label to compute the loss
                target = target_batch[:, di]  # [batch_size]
                target_cluster, target_word = self.cfsm.batch_vocab2cluster(target, self.vocab)
                        
                final_c_dist, final_w_dist, s_t_1, c_t_1 = self.model.decoder(y_t_1, s_t_1, encoder_outputs, 
                                                                              encoder_feature, enc_padding_mask, 
                                                                              c_t_1, target_cluster, di)
                
                c_gold_probs = torch.gather(final_c_dist, 1, target_cluster.unsqueeze(1))  # [batch_size, 1]
                c_step_log_prob = -torch.log(c_gold_probs + config.eps)
                w_gold_probs = torch.gather(final_w_dist, 1, target_word.unsqueeze(1))  # [batch_size, 1]
                w_step_log_prob = -torch.log(w_gold_probs + config.eps)
                step_log_prob = c_step_log_prob + w_step_log_prob
                step_loss = step_log_prob.squeeze() * dec_padding_mask[:, di]
                step_losses.append(step_loss)
                
        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        
        return loss.item()

    def run_eval(self):
        batch = self.batcher.next_batch()
        loss_list = []
        while batch is not None:
            loss = self.eval_one_batch(batch)
            loss_list.append(loss)
            batch = self.batcher.next_batch()
        return np.mean(loss_list)