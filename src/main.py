import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
from torch.autograd import Variable
from progress.bar import Bar
import data
import model
import os
import random


###############################################################################
# Load data
###############################################################################

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)  # [nbatch * bsz, seq_len]
    data = data.t()  # [seq_len, nbatch * bsz]
    return data

def build_dictionary(vocab_path, tag_path):
    word2idx = {}
    tag2idx = {}
    idx2tag = []
    idx2word = []
    word2idx['<pad>'] = 0
    tag2idx['<PAD>'] = 0
    idx2word.append('<pad>')
    idx2tag.append('<PAD>')
    word2idx['<sos>'] = 1
    tag2idx['<SOS>'] = 1
    idx2word.append('<sos>')
    idx2tag.append('<SOS>')
    word2idx['<eos>'] = 2
    tag2idx['<EOS>'] = 2
    idx2word.append('<eos>')
    idx2tag.append('<EOS>')

    with open(vocab_path, 'r') as f:
        words = f.read().split('\n')
    for word in words:
        word = word.strip()
        if word not in word2idx:
            idx2word.append(word)
            word2idx[word] = len(idx2word) - 1

    with open(tag_path, 'r') as f:
        tags = f.read().split('\n')
    for tag in tags:
        tag = tag.strip()
        if tag not in tag2idx:
            idx2tag.append(tag)
            tag2idx[tag] = len(idx2tag) - 1
    
    return word2idx, tag2idx, idx2word, idx2tag

###############################################################################
# Complexity measures
###############################################################################

def get_entropy(o):
    ## o should be a vector scoring possible classes
    probs = nn.functional.softmax(o,dim=0)
    logprobs = nn.functional.log_softmax(o,dim=0) #numerically more stable than two separate operations
    return -1 * torch.sum(probs * logprobs)

def get_surps(o):
    ## o should be a vector scoring possible classes
    logprobs = nn.functional.log_softmax(o,dim=0)
    return -1 * logprobs

def get_guesses(o, scores=False):
    ## o should be a vector scoring possible classes
    guessvals, guessixes = torch.topk(o,args.guessn,0)
    # guessvals are the scores of each input cell
    # guessixes are the indices of the max cells
    if scores:
        return guessvals
    else:
        return guessixes

def get_guessscores(o):
    return get_guesses(o,True)

def get_complexity_iter(o,t):
    for corpuspos,targ in enumerate(t):
        word = corpus.dictionary.idx2word[targ]
        surp = get_surps(o[corpuspos])
        H = get_entropy(o[corpuspos])
        print(str(word)+' '+str(surp)+' '+str(H))

def get_complexity_apply(o, t, sentid, tags=False):
    ## Use apply() method
    Hs = torch.squeeze(apply(get_entropy,o))
    surps = apply(get_surps,o)
    
    if args.guess:
        guesses = apply(get_guesses, o)
        guessscores = apply(get_guessscores, o)
    ## Use dimensional indexing method
    ## NOTE: For some reason, this doesn't work.
    ##       May marginally speed things if we can determine why
    ##       Currently 'probs' ends up equivalent to o after the softmax
    #probs = nn.functional.softmax(o,dim=0)
    #logprobs = nn.functional.log_softmax(o,dim=0)
    #Hs = -1 * torch.sum(probs * logprobs),dim=1)
    #surps = -1 * logprobs
    ## Move along
    for corpuspos,targ in enumerate(t):
        if tags:
            word = corpus.dictionary.idx2tag[int(targ)]
        else:
            word = corpus.dictionary.idx2word[int(targ)]
        if word == '<eos>' or word == '<EOS>':
            #don't output the complexity of EOS
            continue
        surp = surps[corpuspos][int(targ)]
        if args.guess:
            outputguesses = []
            for g in range(args.guessn):
                if tags:
                    outputguesses.append(corpus.dictionary.idx2tag[int(guesses[corpuspos][g])])
                else:
                    outputguesses.append(corpus.dictionary.idx2word[int(guesses[corpuspos][g])])
                if args.guessscores:
                    ##output raw scores
                    outputguesses.append("{:.3f}".format(float(guessscores[corpuspos][g])))
                elif args.guessratios:
                    ##output scores (ratio of score(x)/score(best guess)
                    outputguesses.append("{:.3f}".format(float(guessscores[corpuspos][g])/float(guessscores[corpuspos][0])))
                elif args.guessprobs:
                  ##output probabilities ## Currently normalizes probs over N-best list; ideally it'd normalize to probs before getting the N-best
                  outputguesses.append("{:.3f}".format(math.exp(float(nn.functional.log_softmax(guessscores[corpuspos],dim=0)[g]))))
            outputguesses = ' '.join(outputguesses)
            print(str(word)+' '+str(sentid)+' '+str(corpuspos)+' '+str(len(word))+' '+str(float(surp))+' '+str(float(Hs[corpuspos]))+' '+str(outputguesses))
        else:
            print(str(word)+' '+str(sentid)+' '+str(corpuspos)+' '+str(len(word))+' '+str(float(surp))+' '+str(float(Hs[corpuspos])))

def apply(func, M):
    ## applies a function along a given dimension
    tList = [func(m) for m in torch.unbind(M,dim=0) ]
    res = torch.stack(tList)
    return res

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def get_batch(lm_data, tag_data, i, evaluation=False):
    seq_len = min(args.bptt, lm_data.size(0))
    input_token = Variable(lm_data[:seq_len-1, i:(i+args.batch_size)], volatile=evaluation)
    output_token = Variable(lm_data[1:seq_len, i:(i+args.batch_size)])
    input_tag = Variable(tag_data[:seq_len-1, i:(i+args.batch_size)], volatile=evaluation)
    output_tag = Variable(tag_data[1:seq_len, i:(i+args.batch_size)])
    #This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return input_token.cuda(), output_token.cuda(), input_tag.cuda(), output_tag.cuda()
    else:
        return input_token, output_token, input_tag, output_tag

def test_get_batch(sent_ids, tag_ids, evaluation=True):
    seq_len = sent_ids.size(0)
    input_token = Variable(sent_ids[:seq_len-1], volatile=evaluation)
    output_token = Variable(sent_ids[1:seq_len])
    input_tag = Variable(tag_ids[:seq_len-1], volatile=evaluation)
    output_tag = Variable(tag_ids[1:seq_len])
    # This is where data should be CUDA-fied to lessen OOM errors
    if args.cuda:
        return input_token.cuda(), output_token.cuda(), input_tag.cuda(), output_tag.cuda()
    else:
        return input_token, output_token, input_tag, output_tag

def test_evaluate(args, test_lm_sentences, lm_data_source, ccg_data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    '''
    if args.words:
        print('word sentid sentpos wlen surp entropy')#,end='')
        if args.guess:
            for i in range(args.guessn):
                print(' guess'+str(i))#,end='')
                if args.guessscores:
                    print(' gscore'+str(i))#,end='')
        sys.stdout.write('\n')
    '''
    
    bar = Bar('Processing', max=len(lm_data_source))
    for i in range(len(lm_data_source)):
        sent = test_lm_sentences[i]
        sent_ids = lm_data_source[i]
        tag_ids = ccg_data_source[i]
        if args.cuda:
            sent_ids = sent_ids.cuda()
            tag_ids = tag_ids.cuda()
        if (not args.single) and (torch.cuda.device_count() > 1):
            # "module" is necessary when using DataParallel
            hidden = model.module.init_hidden(1) # number of parallel sentences being processed
        else:
            hidden = model.init_hidden(1) # number of parallel sentences being processed
        
        input_tokens, output_tokens, input_tags, output_tags = test_get_batch(sent_ids, tag_ids, evaluation=True)
        input_tokens = input_tokens.unsqueeze(1)  # [seq_len, 1] 
        input_tags = input_tags.unsqueeze(1)  # [seq_len, 1]
        
        curr_loss = 0
        for t in range(min(args.bptt-1, train_lm_data.size(0)-1)):
            input_token = input_tokens[t].unsqueeze(0) # [1, 1]
            input_tag = input_tags[t].unsqueeze(0) # [1, 1]
            p_word, p_tag, hidden = model(input_token, input_tag, hidden) # p_word = [1, ntoken]
            word_loss = criterion(p_word, output_tokens[t])
            tag_loss = criterion(p_tag, output_tags[t])
            curr_loss += word_loss + tag_loss
        
        total_loss += curr_loss
        '''
        if args.words:
            # output word-level complexity metrics
            get_complexity_apply(output_flat, output_token, i, tags=True)
        else:
            # output sentence-level loss
            print(str(sent)+":"+str(curr_loss))
        '''
        hidden = repackage_hidden(hidden)
        bar.next()
    bar.finish()
    return total_loss / len(lm_data_source)

def evaluate(args, valid_lm_data, valid_ccg_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if (not args.single) and (torch.cuda.device_count() > 1):
        #"module" is necessary when using DataParallel
        hidden = model.module.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)
        
    for i in range(0, valid_lm_data.size(1), args.batch_size):
        if (i+1)*args.batch_size > valid_lm_data.size(1):
            continue
        input_tokens, output_tokens, input_tags, output_tags = get_batch(valid_lm_data, valid_ccg_data, i)
        
        batch_loss = 0
        for t in range(min(args.bptt-1, train_lm_data.size(0)-1)):
            input_token = input_tokens[t].unsqueeze(0)
            input_tag = input_tags[t].unsqueeze(0)
            p_word, p_tag, hidden = model(input_token, input_tag, hidden) # p_word = [batch_size, ntoken]
            word_loss = criterion(p_word, output_tokens[t])
            tag_loss = criterion(p_tag, output_tags[t])
            batch_loss += word_loss + tag_loss
        
        total_loss += batch_loss
        hidden = repackage_hidden(hidden)
    return total_loss / valid_lm_data.size(1)

def train(args, train_lm_data, train_ccg_data):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    if (not args.single) and (torch.cuda.device_count() > 1):
        # "module" is necessary when using DataParallel
        hidden = model.module.init_hidden(args.batch_size)
    else:
        hidden = model.init_hidden(args.batch_size)

    order = list(enumerate(range(0, train_lm_data.size(1), args.batch_size)))
    for batch, i in order:
        if i + args.batch_size >= train_lm_data.size(1):
            continue
        input_tokens, output_tokens, input_tags, output_tags = get_batch(train_lm_data, train_ccg_data, i)
            
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        batch_loss = 0
        for t in range(min(args.bptt-1, train_lm_data.size(0)-1)):
            input_token = input_tokens[t].unsqueeze(0)  # [batch_size, 1]
            input_tag = input_tags[t].unsqueeze(0)  # [batch_size, 1]
            p_word, p_tag, hidden = model(input_token, input_tag, hidden) # p_word = [batch_size, ntoken]
            word_loss = criterion(p_word, output_tokens[t])
            tag_loss = criterion(p_tag, output_tags[t])
            batch_loss += word_loss + tag_loss
        batch_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += batch_loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | {:5.2f} ms/batch | '
                    'loss {:5.2f} '.format(
                epoch, batch, train_lm_data.size(1) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss))
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
    
    parser.add_argument('--vocab_path', type=str, default='../data/vocab.txt',
                        help='location of the language modeling corpus')
    parser.add_argument('--tag_path', type=str, default='../data/tag.txt',
                        help='location of the CCG corpus')
    parser.add_argument('--lm_data', type=str, default='../data/txt',
                        help='location of the language modeling corpus')
    parser.add_argument('--tag_data', type=str, default='../data/tag',
                        help='location of the CCG corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='../models/model.pt',
                        help='path to save the final model')
    parser.add_argument('--single', action='store_true',
                        help='use only a single GPU (even if more are available)')
    parser.add_argument('--save_lm_data', type=str, default='../models/lm_data.bin',
                        help='path to save the LM data')
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
            
    # Load data
    word2idx, tag2idx, idx2word, idx2tag = build_dictionary(args.vocab_path, args.tag_path)
    
    # Build model
    if not args.test:
        ntokens = len(idx2word)
        ntags = len(idx2tag)
        print('Number of unique words:', ntokens)
        print('Number of unique tags:', ntags)
        print('Build model!!!')
        model = model.RNNModel(args.model, ntokens, ntags, args.emsize, args.nhid, args.nlayers, args.dropout)
        if args.cuda:
            if (not args.single) and (torch.cuda.device_count() > 1):
                # Scatters minibatches (in dim=1) across available GPUs
                model = nn.DataParallel(model, dim=1)
            model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    best_val_loss = None
    
    # At any point you can hit Ctrl + C to break out of training early.
    if not args.test:
        try:
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
                train_fname = random.choice(train_files)
                valid_fname = random.choice(valid_files)
                corpus = data.SentenceCorpus(args.bptt, args.lm_data, args.tag_data, 
                                             word2idx, tag2idx, idx2word, idx2tag,
                                             train_fname, valid_fname, None,
                                             save_to=args.save_lm_data, testflag=args.test)

                train_lm_data = batchify(corpus.train_lm, args.batch_size)
                val_lm_data = batchify(corpus.valid_lm, args.batch_size)
                train_ccg_data = batchify(corpus.train_tag, args.batch_size)
                val_ccg_data = batchify(corpus.valid_tag, args.batch_size)
                print('Number of training data:', train_lm_data.size(1))
                print('Number of training tag:', train_ccg_data.size(1))
                print('Number of validation data:', val_lm_data.size(1))
                print('Number of validation tag:', val_ccg_data.size(1))
                    
                epoch_start_time = time.time()
                train(args, train_lm_data, train_ccg_data)
                val_loss = evaluate(args, val_lm_data, val_ccg_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} '.format(epoch, 
                      (time.time() - epoch_start_time), val_loss))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                        best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
    else:
        # Load the best saved model.
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        
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
                                         None, None, test_fname,
                                         save_to=args.save_lm_data, testflag=args.test)
            
            test_lm_sentences, test_lm_data = corpus.test_lm
            test_ccg_data = corpus.test_tag
            
            # Run on test data.
            test_loss += test_evaluate(args, test_lm_sentences, test_lm_data, test_ccg_data)
        print('=' * 89)
        print('| End of testing | test loss {:5.2f} '.format(test_loss))
        print('=' * 89)
