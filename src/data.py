import os
import torch
import dill

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
        self.word2idx['<pad>'] = 0
        self.tag2idx['<PAD>'] = 0
        self.idx2word.append('<pad>')
        self.idx2tag.append('<PAD>')
        self.word2idx['<sos>'] = 1
        self.tag2idx['<SOS>'] = 1
        self.idx2word.append('<sos>')
        self.idx2tag.append('<SOS>')
        self.word2idx['<eos>'] = 2
        self.tag2idx['<EOS>'] = 2
        self.idx2word.append('<eos>')
        self.idx2tag.append('<EOS>')

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
    def __init__(self, seq_len, lm_path, tag_path, save_to='lm_data.bin', testflag=False):
        self.seq_len = seq_len
        if not testflag:
            self.dictionary = Dictionary()
            self.train_lm = self.tokenize(lm_path, 'train')
            self.valid_lm = self.tokenize_with_unks(lm_path, 'valid')
            self.train_tag = self.tokenize_tag(tag_path, 'train')
            self.valid_tag = self.tokenize_tag_with_unks(tag_path, 'valid')
            self.save_to = self.save_dict(save_to)
        else:
            self.dictionary = self.load_dict(save_to)
            self.test_lm = self.sent_tokenize_with_unks(lm_path, 'test')
            self.test_tag = self.sent_tokenize_tag_with_unks(tag_path, 'test')

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

    def tokenize_tag(self, path, mode):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        files = os.listdir(path)
        total_num = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                # Add words and tags to the dictionary
                with open(fpath, 'r') as f:
                    lines = f.read().split('\n')
                total_num += len(lines)
                for line in lines:
                    if line.strip() != "":
                        tags = line.strip().split()
                        for tag in tags:
                            self.dictionary.add_tag(tag)
                    
        # Tokenize file content
        tag_ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        idx = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for i, line in enumerate(lines):
                    if line.strip() != "":
                        tags = line.strip().split()
                        tag_ids[idx, 0] = self.dictionary.tag2idx['<SOS>']
                        for j, tag in enumerate(tags[:self.seq_len-1]):
                            tag_ids[idx, j+1] = self.dictionary.tag2idx[tag]
                        if j+1 < self.seq_len-1:
                            tag_ids[idx, j+2] = self.dictionary.tag2idx['EOS']
                        idx += 1
        return tag_ids

    def tokenize_tag_with_unks(self, path, mode):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        files = os.listdir(path)
        total_num = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                # Add words and tags to the dictionary
                with open(fpath, 'r') as f:
                    lines = f.read().split('\n')
                total_num += len(lines)
                
        # Tokenize file content
        tag_ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        idx = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for i, line in enumerate(lines):
                    if line.strip() != "":
                        tags = line.strip().split()
                        tag_ids[idx, 0] = self.dictionary.tag2idx['<SOS>']
                        for j, tag in enumerate(tags[:self.seq_len-1]):
                            if tag not in self.dictionary.tag2idx:
                                tag_ids[idx, j+1] = self.dictionary.add_tag["<UNK>"]
                            else:
                                tag_ids[idx, j+1] = self.dictionary.tag2idx[tag]
                        if j+1 < self.seq_len-1:
                            tag_ids[idx, j+2] = self.dictionary.tag2idx['EOS']
                        idx += 1
        return tag_ids
    
    def sent_tokenize_tag_with_unks(self, path, mode):
        """Tokenizes and gets POS tags for a text file."""
        assert os.path.exists(path)
        files = os.listdir(path)
        total_num = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                # Add words and tags to the dictionary
                with open(fpath, 'r') as f:
                    lines = f.read().split('\n')
                total_num += len(lines)
        
        # Tokenize file content
        all_tags = []
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for i, line in enumerate(lines):
                    if line.strip() != "":
                        tags = line.strip().split()
                        tag_ids = torch.LongTensor(len(tags)+2)
                        tag_ids[0] = self.dictionary.tag2idx['<SOS>']
                        for j, tag in enumerate(tags):
                            if tag not in self.dictionary.tag2idx:
                                tag_ids[j+1] = self.dictionary.add_tag["<UNK>"]
                            else:
                                tag_ids[j+1] = self.dictionary.tag2idx[tag]
                        tag_ids[j+2] = self.dictionary.tag2idx['<EOS>']
                        all_tags.append(tag_ids)
        return all_tags

    def tokenize(self, path, mode):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        files = os.listdir(path)
        total_num = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                # Add words to the dictionary
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                total_num += len(lines)
                for line in lines:
                    if line.strip() == "":
                        continue
                    words = line.strip().split()
                    for word in words:
                        self.dictionary.add_word(word)

        # Tokenize file content
        ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        idx = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    words = line.strip().split()
                    ids[idx, 0] = self.dictionary.word2idx['<sos>']
                    for j, word in enumerate(words[:self.seq_len-1]):
                        ids[idx, j+1] = self.dictionary.word2idx[word]
                    if j+1 < self.seq_len-1:
                        ids[idx, j+2] = self.dictionary.word2idx['<eos>']
                    idx += 1
        return ids

    def tokenize_with_unks(self, path, mode):
        """Tokenizes a text file, adding unks if needed."""
        assert os.path.exists(path)
        files = os.listdir(path)
        total_num = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                # Add words to the dictionary
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                total_num += len(lines)
                
        # Tokenize file content
        ids = torch.zeros((total_num, self.seq_len), dtype=torch.long)
        idx = 0
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        continue
                    words = line.strip().split()
                    ids[idx, 0] = self.dictionary.word2idx['<sos>']
                    for j, word in enumerate(words[:self.seq_len-1]):
                        if word not in self.dictionary.word2idx:
                            ids[idx, j+1] = self.dictionary.add_word("<unk>")
                        else:
                            ids[idx, j+1] = self.dictionary.word2idx[word]
                    if j+1 < self.seq_len-1:
                        ids[idx, j+2] = self.dictionary.word2idx['<eos>']
                    idx += 1
        return ids

    def sent_tokenize_with_unks(self, path, mode):
        """Tokenizes a text file into sentences, adding unks if needed."""
        assert os.path.exists(path)
        all_ids = []
        sents = []
        files = os.listdir(path)
        for file in files:
            fname = file.split('_')
            if fname[0] == mode:
                fpath = os.path.join(path, file)
                print('Processing file: '+fpath)
                with open(fpath, 'r+', encoding="utf-8") as f:
                    lines = f.read().split('\n')
                for line in lines:
                    if line.strip() == "":
                        continue
                    sents.append(line.strip())
                    words = line.strip().split()
                    # tokenize file content
                    ids = torch.LongTensor(len(words)+2)
                    ids[0] = self.dictionary.word2idx['<sos>']
                    for j, word in enumerate(words):
                        if word not in self.dictionary.word2idx:
                            ids[j+1] = self.dictionary.add_word("<unk>")
                        else:
                            ids[j+1] = self.dictionary.word2idx[word]
                    ids[j+2] = self.dictionary.word2idx['<eos>']
                    all_ids.append(ids)                
        return (sents, all_ids)
