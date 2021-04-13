import os, sys
sys.path.append(os.getcwd())
from src.prepro import *
from src.utils import *
import torch 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='?_?')
parser.add_argument("--dataset", type=str, default="AG")
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--ngrams", type=int, default=2)

args = parser.parse_args()
config = load_config(args.config)
oj = os.path.join

# unigram, unigram+ bigram 버전 2개 저장해두기

for dataset in ["DBP", "Yelp_P", "Yelp_F", 'Yah_A', "Amz_F", "Amz_P"]: # dataset 추가할거임
    print("Dataset : {}".format(dataset))
    data =dict()
    train_path, dev_path, test_path = config.path_rawdata[dataset]

    data_pr_path= oj(config.path_preprocessed, dataset)
    if not os.path.exists(data_pr_path):
        os.mkdir(data_pr_path)

    for n in range(1, args.ngrams+1):
        print("{} gram data processed..".format(n))
        gram = str(n)+"_gram"
        data[gram] = dict()
        word2id,train, train_label, dev, dev_label, test, test_label = load_data(dataset, train_path, dev_path, test_path, n)
    
        assert len(train) == len(train_label)
        assert len(test) == len(test_label)
        assert len(dev) == len(dev_label)

        data[gram]["train"]=train
        data[gram]["train_label"] = train_label
        data[gram]["dev"] = dev
        data[gram]["dev_label"] = dev_label
        data[gram]["test"] = test
        data[gram]["test_label"] = test_label
          
        # word2id save
        torch.save(word2id, data_pr_path+"/"+gram+".pkl")
    # preprocessed data save
    torch.save(data, data_pr_path+"/data.pkl")

print("Done!")


    
    


