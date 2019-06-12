import os
import torch
import dill
import gzip

class convertvocab(object):
    def __init__(self, load_from, save_to):
        self.dictionary = Dictionary()
        self.loadme = self.load_dict(load_from)
        self.save_to = self.save_dict(save_to)

    def save_dict(self, path):
        with open(path, 'wb') as f:
            torch.save(self.dictionary, f, pickle_module=dill)

    def load_dict(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            for line in f:
                self.dictionary.add_word(line.strip())

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = []
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_tag(self, tag):
        if tag not in self.tag2idx:
            self.idx2tag.append(tag)
            self.tag2idx[tag] = len(self.idx2tag) - 1
        return self.tag2idx[tag]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    def __init__(self, seq_len, lm_path, tag_path, save_to='lm_data.bin', testflag=False,
                 trainfname, validfname, testfname):
        self.seq_len = seq_len
        if not testflag:
            self.dictionary = Dictionary()
            self.train_lm = self.tokenize(os.path.join(lm_path, trainfname))
            self.valid_lm = self.tokenize_with_unks(os.path.join(lm_path, validfname))
            self.train_tag = self.tokenize_tag(os.path.join(tag_path, trainfname))
            self.valid_tag = self.tokenize_tag_with_unks(os.path.join(tag_path, validfname))
            self.save_to = self.save_dict(save_to)
        else:
            self.dictionary = self.load_dict(save_to)
            self.test_lm = self.sent_tokenize_with_unks(os.path.join(lm_path, testfname))
            self.test_tag = self.sent_tokenize_tag_with_unks(os.path.join(tag_path, testfname))

    def save_dict(self, path):
        with open(path, 'wb') as f:
            torch.save(self.dictionary, f, pickle_module=dill)

    def load_dict(self, path):
        assert os.path.exists(path)
        with open(path, 'rb') as f:
            fdata = torch.load(f, pickle_module=dill)
            if type(fdata) == type(()):
                # compatibility with old pytorch LM saving
                return(fdata[3])
            return(fdata)

    def tokenize_tag(self, path):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)

        # Add words and tags to the dictionary
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        for line in lines:
            if line.strip() != "":
                tags = line.strip().split()
                for tag in tags:
                    self.dictionary.add_tag(tag)
                    
        # Tokenize file content
        tag_ids = torch.zeros((len(lines), self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() != "":
                tags = line.strip().split()
                for j, tag in enumerate(tags):
                    tag_ids[i, j] = self.dictionary.tag2idx[tag]
        return tag_ids

    def tokenize_tag_with_unks(self, path):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        # Tokenize file content
        tag_ids = torch.zeros((len(lines), self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() != "":
                tags = line.strip().split()
                for j, tag in enumerate(tags):
                    if tag not in self.dictionary.tag2idx:
                        tag_ids[i, j] = self.dictionary.add_tag["<UNK>"]
                    else:
                        tag_ids[i, j] = self.dictionary.tag2idx[tag]
        return tag_ids
    
    def sent_tokenize_tag_with_unks(self, path):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        # Tokenize file content
        all_tags = []
        for i, line in enumerate(lines):
            if line.strip() != "":
                tags = line.strip().split()
                tag_ids = torch.LongTensor(len(tags))
                for j, tag in enumerate(tags):
                    if tag not in self.dictionary.tag2idx:
                        tag_ids[i, j] = self.dictionary.add_tag["<UNK>"]
                    else:
                        tag_ids[i, j] = self.dictionary.tag2idx[tag]
                all_tags.append(tag_ids)
        return all_tags

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        for line in lines:
            if line.strip() == "":
                continue
            words = line.strip().split()
            for word in words:
                self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        ids = torch.zeros((len(lines), self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            words = line.strip().split()
            for j, word in enumerate(words):
                ids[i, j] = self.dictionary.word2idx[word]
        return ids

    def tokenize_with_unks(self, path):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        # Tokenize file content
        with open(path, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        ids = torch.zeros((len(lines), self.seq_len), dtype=torch.long)
        for i, line in enumerate(lines):
            if line.strip() == "":
                continue
            words = line.strip().split()
            for j, word in enumerate(words):
                if word not in self.dictionary.word2idx:
                    ids[i, j] = self.dictionary.add_word("<unk>")
                else:
                    ids[i, j] = self.dictionary.word2idx[word]
        return ids

    def sent_tokenize_with_unks(self, path):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        with open(path, 'r+', encoding="utf-8") as f:
            lines = f.read().split('\n')
        all_ids = []
        sents = []
        for line in lines:
            if line.strip() == "":
                continue
            sents.append(line.strip())
            words = line.strip().split()
            # tokenize file content
            ids = torch.LongTensor(len(words))
            for j, word in enumerate(words):
                if word not in self.dictionary.word2idx:
                    ids[j] = self.dictionary.add_word("<unk>")
                else:
                    ids[j] = self.dictionary.word2idx[word]
                token += 1
            all_ids.append(ids)                
        return (sents, all_ids)
