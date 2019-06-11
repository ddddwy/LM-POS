import os
import argparse
import stanfordnlp

def remove_special_tokens(line):
    new_line = ''
    segs = line.split()
    for seg in segs:
        if seg == '<eos>':
            seg = ''
        if seg == '<unk>':
            seg = 'UNK'
        new_line += seg + ' '
    return new_line

def gen_pos_tags(input_file, out_file):
    nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
    with open(input_file, 'r') as f_in:
        lines = f_in.read().split('\n')
    
    print('Start processing file: '+out_file)
    f_out = open(out_file, 'w')
    for i, line in enumerate(lines):
        new_line = remove_special_tokens(line)
        if len(new_line) > 0:
            doc = nlp(new_line)
            tags = ''
            for sent in doc.sentences:
                for word in sent.words:
                    tags += word.xpos + ' '
            f_out.write(tags+'\n')
        if i % 1000 == 0:
            print('Number of sentences proprcessed:', i)
    f_out.close()
    
def gen_data_batch(data, batch_size, fname):
    batch_num = len(data) // batch_size
    for bid in range(batch_num):
        if bid+1 < batch_num:
            batch = data[bid*batch_size: (bid+1)*batch_size]
        else:
            batch = data[bid*batch_size:]
            
        f = open(fname+str(bid)+'.txt', 'w')
        for line in batch:
            f.write(line+'\n')
        f.close()
        
def split_data(data_dir):
    with open('../data/train.txt', 'r') as f1:
        train = f1.read().split('\n')
    print('Number of train data:', len(train))
    with open('../data/valid.txt', 'r') as f2:
        valid = f2.read().split('\n')
    print('Number of valid data:', len(valid))
    with open('../data/test.txt', 'r') as f3:
        test = f3.read().split('\n')
    print('Number of test data:', len(test))
    
    gen_data_batch(train, 10000, data_dir+'/train_')
    gen_data_batch(valid, 10000, data_dir+'/valid_')
    gen_data_batch(test, 10000, data_dir+'/test_')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data script")
    parser.add_argument("-start", type=int, required=True, 
                        help="")
    parser.add_argument("-end", type=int, required=True, 
                        help="")
    parser.add_argument('--data', action='store_true', default=False,
                        help='Split the origin data')
    args = parser.parse_args()
    
    data_dir = '../data/txt'
    tag_dir = '../data/tag'
    paths = [data_dir, tag_dir]
    for p in paths:
        if not os.path.exists(p):
            os.mkdir(p)
    
    if args.data:
        split_data(data_dir)
    
    files = os.listdir(data_dir)
    files = sorted(files)
    print('Total number of files:', len(files))
    selected = files[args.start:args.end]
    for file in selected:
        gen_pos_tags(data_dir+'/'+file, tag_dir+'/'+file)