import os
import torch

class SentenceCorpus(object):
    def __init__(self, seq_len, lm_path, tag_path, 
                 word2idx, tag2idx, idx2word, idx2tag,
                 train_fname, valid_fname, test_fname,
                 save_to='lm_data.bin', testflag=False):
        self.seq_len = seq_len
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2word = idx2word
        self.idx2tag = idx2tag
        if not testflag:
            self.train_lm = self.tokenize_with_unks(lm_path, train_fname)
            self.valid_lm = self.tokenize_with_unks(lm_path, valid_fname)
            self.train_tag = self.tokenize_tag_with_unks(tag_path, train_fname)
            self.valid_tag = self.tokenize_tag_with_unks(tag_path, valid_fname)
        else:
            self.test_lm = self.sent_tokenize_with_unks(lm_path, test_fname)
            self.test_tag = self.sent_tokenize_tag_with_unks(tag_path, test_fname)

    def tokenize_tag(self, path, fname):
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
                    tag_ids[i, j+1] = self.tag2idx[tag]
                if j+1 < self.seq_len-1:
                    tag_ids[i, j+2] = self.tag2idx['<EOS>']
        return tag_ids

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

    def tokenize(self, path, fname):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        fpath = os.path.join(path, fname)
        with open(fpath, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        total_num = len(lines)

        # Tokenize file content
        ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            words = line.strip().split()
            ids[i, 0] = self.word2idx['<sos>']
            for j, word in enumerate(words[:self.seq_len-1]):
                ids[i, j+1] = self.word2idx[word]
            if j+1 < self.seq_len-1:
                ids[i, j+2] = self.word2idx['<eos>']
        return ids

    def tokenize_with_unks(self, path, fname):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        fpath = os.path.join(path, fname)
        with open(fpath, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        total_num = len(lines)
                
        # Tokenize file content
        ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            words = line.strip().split()
            ids[i, 0] = self.word2idx['<sos>']
            for j, word in enumerate(words[:self.seq_len-1]):
                if word not in self.word2idx:
                    ids[i, j+1] = self.word2idx("<unk>")
                else:
                    ids[i, j+1] = self.word2idx[word]
            if j+1 < self.seq_len-1:
                ids[i, j+2] = self.word2idx['<eos>']
        return ids

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
                    ids[j+1] = self.word2idx("<unk>")
                else:
                    ids[j+1] = self.word2idx[word]
            ids[j+2] = self.word2idx['<eos>']
            all_ids.append(ids)                
        return (sents, all_ids)
