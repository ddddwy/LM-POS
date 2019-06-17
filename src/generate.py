import argparse
import os
import torch
from torch.autograd import Variable
from main import build_dictionary

parser = argparse.ArgumentParser(description='Language Model')

# Model parameters.
parser.add_argument('--vocab_path', type=str, default='../data/vocab.txt',
                    help='location of the language modeling corpus')
parser.add_argument('--tag_path', type=str, default='../data/tag.txt',
                    help='location of the CCG corpus')
parser.add_argument('--seq_len', type=int, default=35,
                    help='max seqence length')
parser.add_argument('--checkpoint', type=str, default='../models/model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='../results/generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=10000,
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log_interval', type=int, default=100,
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

# Load data
word2idx, tag2idx, idx2word, idx2tag = build_dictionary(args.vocab_path, args.tag_path)

# Load model from checkpoint
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
if args.cuda:
    model.cuda()
else:
    model.cpu()

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
        word = idx2word[word_idx]
        input_tag.data.fill_(tag_idx)
        tag = idx2word[tag_idx]

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
