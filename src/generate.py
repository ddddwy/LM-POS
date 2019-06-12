###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import os
import torch
from torch.autograd import Variable
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--lm_path', type=str, default='../data/txt',
                    help='location of the data corpus')
parser.add_argument('--tag_path', type=str, default='../data/tag',
                    help='location of the tag corpus')
parser.add_argument('--seq_len', type=int, default=35,
                    help='max seqence length')
parser.add_argument('--checkpoint', type=str, default='../models/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--lm_data', type=str, default='/models/lm_data.bin',
                    help='path to load the LM data')
parser.add_argument('--outf', type=str, default='../results/generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")
    
if not os.path.exists('../results'):
    os.mkdir('../results')

# Load model from checkpoint
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
if args.cuda:
    model.cuda()
else:
    model.cpu()

# Load corpus
corpus = data.SentenceCorpus(args.seq_len, args.lm_path, args.tag_path, args.lm_data, True)

# Initialize model inputs
hidden = model.init_hidden(1)
input_token = Variable(torch.ones(1, 1).long(), volatile=True)
input_tag = Variable(torch.ones(1, 1).long(), volatile=True)
if args.cuda:
    input_token = input_token.cuda()
    input_tag = input_tag.cuda()

with open(args.outf, 'w') as outf:
    for i in range(args.words):
        p_word, p_tag, hidden = model(input_token, input_tag, hidden)
        word_idx = torch.argmax(p_word.squeeze())
        tag_idx = torch.argmax(p_tag.squeeze())
        input_token.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        input_tag.data.fill_(tag_idx)
        tag = corpus.dictionary.idx2word[tag_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
