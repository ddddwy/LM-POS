import argparse
import time
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
from progress.bar import Bar
from data import build_dictionary, batchify
import data
import model
import os
import random
import numpy as np
import tensorflow as tf


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def get_batch(lm_data, lm_masking, tag_data, i, evaluation=False):
    seq_len = min(args.bptt, lm_data.size(0))
    input_token = Variable(lm_data[:seq_len-1, i:(i+args.batch_size)], volatile=evaluation)
    output_token = Variable(lm_data[1:seq_len, i:(i+args.batch_size)])
    input_tag = Variable(tag_data[:seq_len-1, i:(i+args.batch_size)], volatile=evaluation)
    output_tag = Variable(tag_data[1:seq_len, i:(i+args.batch_size)])
    masking = Variable(lm_masking[:seq_len, i:(i+args.batch_size)], volatile=evaluation)
    #This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return input_token.cuda(), output_token.cuda(), input_tag.cuda(), output_tag.cuda(), masking.cuda()
    else:
        return input_token, output_token, input_tag, output_tag, masking

def test_get_batch(sent_ids, tag_ids):
    seq_len = sent_ids.size(0)
    input_token = Variable(sent_ids[:seq_len-1])
    output_token = Variable(sent_ids[1:seq_len])
    input_tag = Variable(tag_ids[:seq_len-1])
    output_tag = Variable(tag_ids[1:seq_len])
    # This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return input_token.cuda(), output_token.cuda(), input_tag.cuda(), output_tag.cuda()
    else:
        return input_token, output_token, input_tag, output_tag

def test_evaluate(args, model, test_lm_sentences, lm_data_source, ccg_data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    bar = Bar('Processing', max=len(lm_data_source))
    for i in range(len(lm_data_source)):
        sent_ids = lm_data_source[i]
        tag_ids = ccg_data_source[i]
        if args.cuda:
            sent_ids = sent_ids.cuda()
            tag_ids = tag_ids.cuda()
            
        with torch.no_grad():
            hidden_word = model.init_hidden(1)
            hidden_tag = model.init_hidden(1)
            input_tokens, output_tokens, input_tags, output_tags = test_get_batch(sent_ids, tag_ids)
            input_tokens = input_tokens.unsqueeze(1)  # [seq_len, 1] 
            input_tags = input_tags.unsqueeze(1)  # [seq_len, 1]
            output_tokens = output_tokens.unsqueeze(1)  # [seq_len, 1]
            output_tags = output_tags.unsqueeze(1)  # [seq_len, 1]
            
            curr_loss = 0
            for t in range(min(sent_ids.size(0)-1, tag_ids.size(0)-1)):
                input_token = input_tokens[t].unsqueeze(0) # [1, 1]
                input_tag = input_tags[t].unsqueeze(0) # [1, 1]
                if args.rnn_num == 1:
                    # p_word = [1, ntoken]
                    if args.lm:
                        p_word, hidden_word = model(input_token, hidden_word)
                    else:
                        p_word, p_tag, hidden_word = model(input_token, input_tag, hidden_word)
                else:
                    p_word, p_tag, hidden_word, hidden_tag = model(input_token, input_tag, hidden_word, hidden_tag)
                word_loss = criterion(p_word, output_tokens[t])
                curr_loss += word_loss[0]
            seq_len = min(sent_ids.size(0), tag_ids.size(0))
            total_loss += float(curr_loss)/seq_len
            hidden_word = repackage_hidden(hidden_word)
            hidden_tag = repackage_hidden(hidden_tag)
        bar.next()
    bar.finish()
    return total_loss / len(lm_data_source)

def evaluate(args, model, valid_lm_data, valid_masking, valid_ccg_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden_word = model.init_hidden(args.batch_size)
    hidden_tag = model.init_hidden(args.batch_size)
        
    for i in range(0, valid_lm_data.size(1), args.batch_size):
        if (i+1)*args.batch_size > valid_lm_data.size(1):
            continue
        input_tokens, output_tokens, input_tags, output_tags, masking = get_batch(valid_lm_data, valid_masking, 
                                                                                  valid_ccg_data, i)
        
        seq_loss = 0
        for t in range(min(args.bptt-1, valid_lm_data.size(0)-1)):
            input_token = input_tokens[t].unsqueeze(0)
            input_tag = input_tags[t].unsqueeze(0)
            if args.rnn_num == 1:
                # p_word = [batch_size, ntoken]
                if args.lm:
                    p_word, hidden_word = model(input_token, hidden_word)
                else:
                    p_word, p_tag, hidden_word = model(input_token, input_tag, hidden_word) 
            else:
                p_word, p_tag, hidden_word, hidden_tag = model(input_token, input_tag, hidden_word, hidden_tag)
            word_loss = criterion(p_word, output_tokens[t]) # [batch_size]
            word_loss = word_loss * masking[t] 
            per_word_loss = torch.mean(word_loss)
            if args.lm:
                seq_loss += per_word_loss
            else:
                tag_loss = criterion(p_tag, output_tags[t])
                tag_loss = tag_loss * masking[t]
                per_tag_loss = torch.mean(tag_loss)
                seq_loss += per_word_loss + per_tag_loss
        
        seq_len = min(args.bptt-1, valid_lm_data.size(0)-1)
        total_loss += float(seq_loss)/seq_len
        
        hidden_word = repackage_hidden(hidden_word)
        hidden_tag = repackage_hidden(hidden_tag)
    return total_loss/(valid_lm_data.size(1)//args.batch_size)

def train(args, model, train_lm_data, train_masking, train_ccg_data, criterion, optimizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden_word = model.init_hidden(args.batch_size)
    hidden_tag = model.init_hidden(args.batch_size)

    order = list(enumerate(range(0, train_lm_data.size(1), args.batch_size)))
    for batch, i in order:
        if i + args.batch_size >= train_lm_data.size(1):
            continue
        input_tokens, output_tokens, input_tags, output_tags, masking = get_batch(train_lm_data, train_masking, 
                                                                                  train_ccg_data, i)
            
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden_word = repackage_hidden(hidden_word)
        hidden_tag = repackage_hidden(hidden_tag)
        optimizer.zero_grad()
        seq_loss = 0
        for t in range(min(args.bptt-1, train_lm_data.size(0)-1)):
            input_token = input_tokens[t].unsqueeze(0)  # [1, batch_size]
            input_tag = input_tags[t].unsqueeze(0)  # [1, batch_size]
            if args.rnn_num == 1:
                # p_word = [batch_size, ntoken]
                if args.lm:
                    p_word, hidden_word = model(input_token, hidden_word)
                else:
                    p_word, p_tag, hidden_word = model(input_token, input_tag, hidden_word)
            else:
                p_word, p_tag, hidden_word, hidden_tag = model(input_token, input_tag, hidden_word, hidden_tag)
            word_loss = criterion(p_word, output_tokens[t]) # [batch_size]
            word_loss = word_loss * masking[t] 
            per_word_loss = torch.mean(word_loss)
            if args.lm:
                seq_loss += per_word_loss
            else:
                tag_loss = criterion(p_tag, output_tags[t])
                tag_loss = tag_loss * masking[t]
                per_tag_loss = torch.mean(tag_loss)
                seq_loss += per_word_loss + per_tag_loss
        seq_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        seq_len = min(args.bptt-1, train_lm_data.size(0)-1)
        total_loss += float(seq_loss)/seq_len

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:1.4f} | {:5.2f} ms/batch | '
                    'loss {:5.4f} '.format(epoch, batch, train_lm_data.size(1) // args.batch_size, 
                          lr, elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    ## Parallelization notes:
    ##   Does not currently operate across multiple nodes
    ##   Single GPU is better for default: tied,emsize:200,nhid:200,nlayers:2,dropout:0.2
    ##
    ##   Multiple GPUs are better for tied,emsize:1500,nhid:1500,nlayers:2,dropout:0.65
    ##      4 GPUs train on wikitext-2 in 1/2 - 2/3 the time of 1 GPU
    
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM language modeling and CCG tagging multitask model')
    
    parser.add_argument('--vocab_path', type=str, default='../data/vocab_50k.txt',
                        help='location of the language modeling corpus')
    parser.add_argument('--tag_path', type=str, default='../data/tag.txt',
                        help='location of the CCG corpus')
    parser.add_argument('--lm_data', type=str, default='../data/txt',
                        help='location of the language modeling corpus')
    parser.add_argument('--tag_data', type=str, default='../data/tag',
                        help='location of the CCG corpus')
    parser.add_argument('--rnn_num', type=int, default=1,
                        help='number of recurrent net')
    parser.add_argument('--simple', action='store_true',
                        help='use simple model')
    parser.add_argument('--lm', action='store_true',
                        help='use lm model')
    parser.add_argument('--load', action='store_true',
                        help='load pre-trained model')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=100,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=125, metavar='N',
                        help='report interval')
    parser.add_argument('--exp', type=str,  default='single',
                        help='path to save the final model')
    parser.add_argument('--test', action='store_true',
                        help='test a trained LM')
    parser.add_argument('--guess', action='store_true',
                        help='display best guesses at each time step')
    parser.add_argument('--guessscores', action='store_true',
                        help='display guess scores along with guesses')
    parser.add_argument('--guessratios', action='store_true',
                        help='display guess ratios normalized by best guess')
    parser.add_argument('--guessprobs', action='store_true',
                        help='display guess probs along with guesses')
    parser.add_argument('--guessn', type=int, default=1,
                        help='output top n guesses')
    parser.add_argument('--words', action='store_true',
                        help='evaluate word-level complexities (instead of sentence-level loss)')
    args = parser.parse_args()
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            
        else:
            torch.cuda.manual_seed(args.seed)
            
    if not os.path.exists('../models'):
        os.mkdir('../models')
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    
    eval_log = '../logs/'+args.exp
    if not os.path.exists(eval_log):
        os.mkdir(eval_log)
    summary_writer = tf.summary.FileWriter(eval_log)
            
    # Load data
    word2idx, tag2idx, idx2word, idx2tag = build_dictionary(args.vocab_path, args.tag_path)
    ntokens = len(idx2word)
    ntags = len(idx2tag)
    print('Number of unique words:', ntokens)
    print('Number of unique tags:', ntags)
    
    # Build model
    if not args.test and not args.load:
        print('Build model!!!')
        if args.rnn_num == 1:
            if args.lm:
                model =  model.SimpleLMModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)
            if args.simple:
                model =  model.SimpleRNNModel(args.model, ntokens, ntags, args.emsize, args.nhid, args.nlayers, args.dropout)
            if not args.simple and not args.lm:
                model = model.RNNModel(args.model, ntokens, ntags, args.emsize, args.nhid, args.nlayers, args.dropout)
        else:
            if args.simple:
                model = model.SimpleMultiRNNModel(args.model, ntokens, ntags, args.emsize, args.nhid, args.nlayers, args.dropout)
            else:
                model = model.MultiRNNModel(args.model, ntokens, ntags, args.emsize, args.nhid, args.nlayers, args.dropout)
        if args.cuda:
            model.cuda()
    
    if args.load:
        # Load the best saved model.
        with open('../models/'+args.exp+'.pt', 'rb') as f:
            if args.cuda:
                model = torch.load(f)
            else:
                model = torch.load(f, map_location='cpu')
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    lr = args.lr
    best_val_loss = None
    
    # At any point you can hit Ctrl + C to break out of training early.
    if not args.test:
        try:
            optimizer = torch.optim.SGD(model.parameters(), lr)
            files = os.listdir(args.lm_data)
            files = sorted(files)
            train_files = []
            valid_files = []
            for file in files:
                prefix = file.split('_')[0]
                if prefix == 'train':
                    train_files.append(file)
                if prefix == 'valid':
                    valid_files.append(file)
            print('Start training!!!')
            for epoch in range(1, args.epochs+1):
                valid_fname = random.choice(valid_files)
                for train_fname in train_files:
                    train_fname = random.choice(train_files)
                    corpus = data.SentenceCorpus(args.bptt, args.lm_data, args.tag_data, 
                                                 word2idx, tag2idx, idx2word, idx2tag,
                                                 train_fname, valid_fname, None, testflag=args.test)
    
                    train_lm_data = batchify(corpus.train_lm, args.batch_size)
                    train_masking = batchify(corpus.train_maksing, args.batch_size)
                    train_ccg_data = batchify(corpus.train_tag, args.batch_size)
                    
                    epoch_start_time = time.time()
                    train(args, model, train_lm_data, train_masking, train_ccg_data, criterion, optimizer)

                    val_lm_data = batchify(corpus.valid_lm, args.batch_size)
                    val_masking = batchify(corpus.valid_maksing, args.batch_size)
                    val_ccg_data = batchify(corpus.valid_tag, args.batch_size)
                    val_loss = evaluate(args, model, val_lm_data, val_masking, val_ccg_data)
                    print('-' * 80)
                    print('| end of {} | time: {:5.2f}s | valid loss {:5.4f} '.format(train_fname, 
                          (time.time() - epoch_start_time), val_loss))
                    print('-' * 80)
                    # Save the model if the validation loss is the best we've seen so far.
                    if best_val_loss is None or val_loss < best_val_loss:
                        with open('../models/'+args.exp+'.pt', 'wb') as f:
                            torch.save(model, f)
                            best_val_loss = val_loss
                
                total_val_loss = 0
                valid_name = valid_files[0]
                corpus = data.SentenceCorpus(args.bptt, args.lm_data, args.tag_data, 
                                             word2idx, tag2idx, idx2word, idx2tag,
                                             train_fname, valid_fname, None, testflag=args.test)
                val_lm_data = batchify(corpus.valid_lm, args.batch_size)
                val_masking = batchify(corpus.valid_maksing, args.batch_size)
                val_ccg_data = batchify(corpus.valid_tag, args.batch_size)
                val_loss = evaluate(args, model, val_lm_data, val_masking, val_ccg_data)
                total_val_loss += val_loss
                print('-' * 80)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} '.format(epoch, 
                      (time.time() - epoch_start_time), total_val_loss))
                print('-' * 80)
                
                loss_sum = tf.Summary()
                loss_sum.value.add(tag='val_loss', simple_value=total_val_loss)
                summary_writer.add_summary(loss_sum, global_step=epoch)
                summary_writer.flush()
                
        except KeyboardInterrupt:
            print('-' * 80)
            print('Exiting from training early')
    else:
        # Load the best saved model.
        with open('../models/'+args.exp+'.pt', 'rb') as f:
            if args.cuda:
                model = torch.load(f)
            else:
                model = torch.load(f, map_location='cpu')
        
        files = os.listdir(args.lm_data)
        files = sorted(files)
        test_files = []
        for file in files:
            prefix = file.split('_')[0]
            if prefix == 'test':
                test_files.append(file)
                
        test_loss = 0.
        for test_fname in test_files:
            corpus = data.SentenceCorpus(args.bptt, args.lm_data, args.tag_data, 
                                         word2idx, tag2idx, idx2word, idx2tag,
                                         None, None, test_fname, testflag=args.test)
            
            test_lm_sentences, test_lm_data = corpus.test_lm
            test_ccg_data = corpus.test_tag
            
            # Run on test data.
            curr_loss = test_evaluate(args, model, test_lm_sentences, test_lm_data, test_ccg_data)
            test_loss += curr_loss
            print('| End of testing | test loss {:5.4f} '.format(curr_loss))
        test_loss = test_loss / len(test_files)
        print('=' * 80)
        print('| End of testing | test loss {:5.4f} | test ppl {:8.4f}'.format(test_loss, np.power(2, test_loss)))
        print('=' * 80)
