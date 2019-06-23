import argparse
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from main import build_dictionary

parser = argparse.ArgumentParser(description='Language Model')

# Model parameters.
parser.add_argument('--vocab_path', type=str, default='../data/vocab_50k.txt',
                    help='location of the language modeling corpus')
parser.add_argument('--tag_path', type=str, default='../data/tag.txt',
                    help='location of the CCG corpus')
parser.add_argument('--seq_len', type=int, default=35,
                    help='max seqence length')
parser.add_argument('--rnn_num', type=int, default=1,
                    help='number of recurrent net')
parser.add_argument('--exp', type=str, default='lm',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='../results/generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default=1000,
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
with open('../models/'+args.exp+'.pt', 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f, map_location='cpu')

model.eval()
if args.cuda:
    model.cuda()
else:
    model.cpu()

with torch.no_grad():
    # Initialize model inputs
    hidden_word = model.init_hidden(1)
    hidden_tag = model.init_hidden(1)
    input_token = Variable(3*torch.ones(1, 1).long())
    input_tag = Variable(3*torch.ones(1, 1).long())
    if args.cuda:
        input_token = input_token.cuda()
        input_tag = input_tag.cuda()
    
    with open(args.outf, 'w') as outf:
        true_words = 0
        for i in range(args.words):
            if args.rnn_num == 1:
                if args.lm:
                    p_word, hidden_word = model(input_token, hidden_word)
                else:
                    p_word, p_tag, hidden_word = model(input_token, input_tag, hidden_word)
            else:
                p_word, p_tag, hidden_word, hidden_tag = model(input_token, input_tag, hidden_word, hidden_tag)
            word_dist = F.softmax(p_word.squeeze(), dim=0)
            word_idx = torch.multinomial(word_dist, 2, replacement=False)
            if word_idx[0] == 0:
                wid = word_idx[1]
            else:
                wid = word_idx[0]
            tag_dist = F.softmax(p_tag.squeeze(), dim=0)
            tag_idx = torch.multinomial(tag_dist, 2, replacement=False)
            if tag_idx[0] == 0:
                tid = tag_idx[1]
            else:
                tid = tag_idx[0]
            input_token.data.fill_(wid)
            word = idx2word[wid]
            input_tag.data.fill_(tid)
            tag = idx2word[tid]
            
            if word == '<eos>':
                continue
            else:
                outf.write(word + ' ')
                true_words += 1
                if true_words % 20 == 19:
                    outf.write('\n')
    
            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
