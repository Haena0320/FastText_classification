import re
import numpy as np
import gzip
import pickle

def get_vocab(data_list=None, dataset="", n=1):

    ngram2id = dict()
    for data in data_list: # file path
        #if dataset == "AG": # kind of dataset
        f = open(data, encoding="utf-8")
        ngram2id["padding"] = 0
        id = 1 # padding 0 있음
        for line in f:
            words = line_to_words(line, dataset, n)
            for word in words:
                if word not in ngram2id:
                    ngram2id[word] = id
                    id += 1
        f.close()
    print("{} gram embeddings : {}".format(n ,len(ngram2id.keys())))
    return ngram2id

def line_to_words(line=None, dataset="", n=1):
    clean_line= clean_str(line)
    clean_line = clean_line.split(" ")
    #if dataset =="AG":
    words = clean_line[2:]
    if n > 1:
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-(n-1))]
        words += ngrams
    return words

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(dataset, train_path, dev_path="", test_path='' , n=1):
    f_names = [train_path]
    if not test_path =="": f_names.append(test_path)
    if not dev_path =="": f_names.append(dev_path)

    ngram2id = get_vocab(data_list=f_names, dataset=dataset, n=n)

    train = []
    train_label = []

    dev = []
    dev_label = []

    test=[]
    test_label=[]

    files=[]
    data= []
    data_label = []

    f_train = open(train_path, 'r')
    files.append(f_train)
    data.append(train)
    data_label.append(train_label)

    if not test_path=="":
        f_test = open(test_path, 'r')
        files.append(f_test)
        data.append(test)
        data_label.append(test_label)

    if not dev_path=="":
        f_dev = open(dev_path, "r")
        files.append(f_dev)
        data.append(dev)
        data_abel.append(dev_label)

    for d, l, f in zip(data, data_label, files):

        for line in f:
            ngrams = line_to_words(line, dataset)
            if dataset in ["AG", "Sogou", "Yelp_P", "Yelp_F", 'Yah_A', "Amz_F", "Amz_P"]:
                y = int(line.strip().split(",")[0][1])-1
            elif dataset=="DBP":
                y = int(line.strip().split(",")[0]) - 1

            sent = [ngram2id[word] for word in ngrams]
            d.append(sent)
            l.append(y)
    f_train.close()
    if not test_path == "":
        f_test.close()
    if not dev_path == "":
        f_dev.close()

    # data adjust
    train, train_label, dev, dev_label, test, test_label = adjust_data(train, train_label, dev, dev_label, test, test_label)
    return ngram2id, np.array(train, dtype=object), np.array(train_label, dtype=object),np.array(dev, dtype=object), np.array(dev_label, dtype=object), np.array(test, dtype=object), np.array(test_label, dtype=object)

def adjust_data(train, train_label, test, test_label, dev, dev_label):
    limit = len(train)//10
    if (len(test)==0 & len(dev)==0):
        test = train[:limit]
        test_label = train_label[:limit]

        dev = train[limit:limit*2]
        dev_label = train_label[limit:limit*2]

    elif (len(test)==0 & len(dev)!=0):
        test = train[:limit]
        test_label = train_label[:limit]
    elif (len(test)!=0 & len(dev)==0):
        dev = train[:limit]
        dev_label = train_label[:limit]

    else:
        pass
    return train, train_label, dev, dev_label, test, test_label


# with open("/hdd1/user15/workspace/FastText/data/raw/dbpedia_csv/train.csv", "r") as f:
#     line = f.readline()
#
# int(line.strip().split(",")[0])-1
# int(line.strip().split(",")[0])-1