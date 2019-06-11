#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
import os
import time

import torch
from torch.autograd import Variable
import nltk
import numpy as np

from batcher import Batcher
from data import Vocab
import data, config
from model import Model
from utils import write_for_rouge, rouge_eval, rouge_log
from train_util import get_input_from_batch
from cfsm import CFSM

use_cuda = config.use_gpu and torch.cuda.is_available()


class GreedySearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(5)
        
        self.cfsm = CFSM()
        self.model = Model(model_file_path, is_eval=True)


    def decode(self):
        start = time.time()
        counter = 0
        bleu_scores = []
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run greedy search to get best Hypothesis
            best_summary = self.greedy_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = []
            for t in best_summary:
                output_ids.append(int(t))
            decoded_words = data.outputids2words(output_ids, self.vocab, None)

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstracts = batch.original_abstracts_sents[0]
#            print('Q1: ')
#            print(batch.original_articles[0])
#            print('Q2: ')
#            print(original_abstracts[0])
#            print('Decoded Q2: ')
#            print(' '.join(decoded_words))
#            print('++++++++++++++++++++++++++++++++++++++++++')
            
            reference = original_abstracts[0].strip().split()
            bleu = nltk.translate.bleu_score.sentence_bleu([reference], decoded_words, weights = (0.5, 0.5))
            bleu_scores.append(bleu)
            
            write_for_rouge(original_abstracts, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()
            
        print(np.mean(bleu_scores))


    def greedy_search(self, batch):
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, c_t_1 = get_input_from_batch(batch, use_cuda)
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        results = []
        steps = 0
        latest_batch = Variable(torch.LongTensor(config.batch_size*[2]))
        if use_cuda:
            latest_batch = latest_batch.cuda()
        
        while steps < config.max_dec_steps:
            y_t_1 = latest_batch
            
            # compute the next state
            final_c_dist, final_w_dist, s_t_1, c_t_1 = self.model.decoder(y_t_1, s_t_1, encoder_outputs, 
                                                                          encoder_feature, enc_padding_mask, 
                                                                          c_t_1, None, steps)

            cid_batch = torch.argmax(final_c_dist, dim=1) # [batch_size]
            cwid_batch = torch.argmax(final_w_dist, dim=1) # [batch_size]
            latest_batch, output_batch = self.cfsm.batch_decode(cid_batch, cwid_batch, self.vocab)
            steps += 1
            
            results.append(output_batch)

        final_results = torch.stack(results, 1)
        final_results = final_results[0].detach().cpu().numpy()
        
        return final_results

if __name__ == '__main__':
	model_filename = sys.argv[1]
	beam_Search_processor = GreedySearch(model_filename)
	beam_Search_processor.decode()
    
#	decode_dir = sys.argv[1]
#	rouge_ref_dir = os.path.join(decode_dir, "rouge_ref")
#	rouge_dec_dir = os.path.join(decode_dir, "rouge_dec_dir")
#	results_dict = rouge_eval(rouge_ref_dir, rouge_dec_dir)
#	rouge_log(results_dict, decode_dir)