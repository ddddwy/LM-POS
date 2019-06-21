import os
import torch

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
    word2idx['<unk>'] = 1
    tag2idx['<UNK>'] = 1
    idx2word.append('<unk>')
    idx2tag.append('<UNK>')
    word2idx['<sos>'] = 2
    tag2idx['<SOS>'] = 2
    idx2word.append('<sos>')
    idx2tag.append('<SOS>')
    word2idx['<eos>'] = 3
    tag2idx['<EOS>'] = 3
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

class SentenceCorpus(object):
    def __init__(self, seq_len, lm_path, tag_path, word2idx, tag2idx, idx2word, idx2tag,
                 train_fname, valid_fname, test_fname, testflag=False):
        self.seq_len = seq_len
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        if not testflag:
            self.train_lm, self.train_maksing = self.tokenize_with_unks(lm_path, train_fname)
            self.valid_lm, self.valid_maksing = self.tokenize_with_unks(lm_path, valid_fname)
            self.train_tag = self.tokenize_tag_with_unks(tag_path, train_fname)
            self.valid_tag = self.tokenize_tag_with_unks(tag_path, valid_fname)
        else:
            self.test_lm = self.sent_tokenize_with_unks(lm_path, test_fname)
            self.test_tag = self.sent_tokenize_tag_with_unks(tag_path, test_fname)

    def tokenize_tag_with_unks(self, path, fname):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        fpath = os.path.join(path, fname)
        with open(fpath, 'r') as f:
            lines = f.read().split('\n')
        total_num = len(lines)
                
        # Tokenize file content
        tag_ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() != "":
                tags = line.strip().split()
                tag_ids[i, 0] = self.tag2idx['<SOS>']
                for j, tag in enumerate(tags[:self.seq_len-1]):
                    if tag not in self.tag2idx:
                        tag_ids[i, j+1] = self.tag2idx["<UNK>"]
                    else:
                        tag_ids[i, j+1] = self.tag2idx[tag]
                if j+1 < self.seq_len-1:
                    tag_ids[i, j+2] = self.tag2idx['<EOS>']
        return tag_ids
    
    def sent_tokenize_tag_with_unks(self, path, fname):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        fpath = os.path.join(path, fname)
        with open(fpath, 'r') as f:
            lines = f.read().split('\n')
        
        # Tokenize file content
        all_tags = []
        for i, line in enumerate(lines):
            if line.strip() != "":
                tags = line.strip().split()
                tag_ids = torch.LongTensor(len(tags)+2)
                tag_ids[0] = self.tag2idx['<SOS>']
                for j, tag in enumerate(tags):
                    if tag not in self.tag2idx:
                        tag_ids[j+1] = self.tag2idx["<UNK>"]
                    else:
                        tag_ids[j+1] = self.tag2idx[tag]
                tag_ids[j+2] = self.tag2idx['<EOS>']
                all_tags.append(tag_ids)
        return all_tags

    def tokenize_with_unks(self, path, fname):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        fpath = os.path.join(path, fname)
        with open(fpath, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        total_num = len(lines)
                
        # Tokenize file content
        ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        ids_masking = torch.zeros((total_num, self.seq_len))
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            words = line.strip().split()
            ids[i, 0] = self.word2idx['<sos>']
            ids_masking[i, 0] = 1
            for j, word in enumerate(words[:self.seq_len-1]):
                ids_masking[i, j+1] = 1
                if word not in self.word2idx:
                    ids[i, j+1] = self.word2idx["<unk>"]
                else:
                    ids[i, j+1] = self.word2idx[word]
        return ids, ids_masking

    def sent_tokenize_with_unks(self, path, fname):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        fpath = os.path.join(path, fname)
        with open(fpath, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        for line in lines:
            if line.strip() == "":
                continue
            sents.append(line.strip())
            words = line.strip().split()
            # tokenize file content
            ids = torch.LongTensor(len(words)+2)
            ids[0] = self.word2idx['<sos>']
            for j, word in enumerate(words):
                if word not in self.word2idx:
                    ids[j+1] = self.word2idx["<unk>"]
                else:
                    ids[j+1] = self.word2idx[word]
            ids[j+2] = self.word2idx['<eos>']
            all_ids.append(ids)                
        return (sents, all_ids)
