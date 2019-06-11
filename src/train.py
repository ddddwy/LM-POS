from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import shutil

import tensorflow as tf
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam

import config
from model import Model
from batcher import Batcher
from data import Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from eval import Evaluate
from cfsm import CFSM

use_cuda = config.use_gpu and torch.cuda.is_available()
tf.logging.set_verbosity(tf.logging.INFO)


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(5)
        
        if not os.path.exists(config.log_root):
            os.mkdir(config.log_root)

        self.model_dir = os.path.join(config.log_root, 'train_model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        self.eval_log = os.path.join(config.log_root, 'eval_log')
        if not os.path.exists(self.eval_log):
            os.mkdir(self.eval_log)
        self.summary_writer = tf.summary.FileWriter(self.eval_log)

    def save_model(self, running_avg_loss, iter, mode):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        if mode == 'train':
            save_model_dir = self.model_dir
        else:
            best_model_dir = os.path.join(config.log_root, 'best_model')
            if not os.path.exists(best_model_dir):
                os.mkdir(best_model_dir)
            save_model_dir = best_model_dir
        
        if len(os.listdir(save_model_dir))>0:
            shutil.rmtree(save_model_dir)
            time.sleep(2)
            os.mkdir(save_model_dir)
        train_model_path = os.path.join(save_model_dir, 'model_best_%d'%(iter))
        torch.save(state, train_model_path)
        return train_model_path
    

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)
        self.cfsm = CFSM()

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        if config.optimizer == 'adam':
            self.optimizer = Adam(params, lr=config.lr)
        else:
            self.optimizer = Adagrad(params, lr=0.15, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

        return start_iter, start_loss
    

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, c_t_1 = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        
        log_probs = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            # teacher forcing
            y_t_1 = dec_batch[:, di]  
            
            # use ground truth word label to compute the loss
            target = target_batch[:, di]  # [batch_size]
            target_cluster, target_word = self.cfsm.batch_vocab2cluster(target, self.vocab)
                
            # compute the next state
            final_c_dist, final_w_dist, s_t_1, c_t_1 = self.model.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, 
                                                                          enc_padding_mask, c_t_1, target_cluster, di)
            
            c_gold_probs = torch.gather(final_c_dist, 1, target_cluster.unsqueeze(1))  # [batch_size, 1]
            c_step_log_prob = -torch.log(c_gold_probs + config.eps)
            w_gold_probs = torch.gather(final_w_dist, 1, target_word.unsqueeze(1))  # [batch_size, 1]
            w_step_log_prob = -torch.log(w_gold_probs + config.eps)
            step_log_prob = c_step_log_prob + w_step_log_prob
            step_loss = step_log_prob.squeeze() * dec_padding_mask[:, di]
            log_probs.append(step_loss)
        
        all_logp = torch.sum(torch.stack(log_probs, 1), 1)  # [batch_size]
        loss = torch.mean(all_logp)
        
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()
    

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        min_val_loss = np.inf
        
        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)
            
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
            iter += 1
            
            if iter % config.save_model_iter == 0:
                model_file_path = self.save_model(running_avg_loss, iter, mode='train')
                evl_model = Evaluate(model_file_path)
                val_avg_loss = evl_model.run_eval()
                if val_avg_loss < min_val_loss:
                    min_val_loss = val_avg_loss
                    best_model_file_path = self.save_model(running_avg_loss, iter, mode='eval')
                    tf.logging.info('Save best model at %s'%best_model_file_path)
                
                tf.logging.info('steps %d, %.2f seconds for %d iters, loss: %f, val_loss: %f' 
                                % (iter, config.save_model_iter, time.time() - start, loss, val_avg_loss))
                start = time.time()
                
                loss_sum = tf.Summary()
                loss_sum.value.add(tag='val_avg_loss', simple_value=val_avg_loss)
                self.summary_writer.add_summary(loss_sum, global_step=iter)
                self.summary_writer.flush()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
