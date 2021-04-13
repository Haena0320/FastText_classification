import sys, os
sys.path.append(os.getcwd())

import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default='?_?')
parser.add_argument("--model", type=str, default="FastText")
parser.add_argument("--dataset", type=str, default='AG')
parser.add_argument("--task", type=str, default="sent")
parser.add_argument("--gpu", type=str, default=None)

parser.add_argument("--h_units",type=int , default=10)
parser.add_argument('--ngram', type=int, default=1)

parser.add_argument("--config", type=str, default="default")
parser.add_argument("--log", type=str, default="log")
parser.add_argument("--epochs", type=int, default=5)

parser.add_argument("--learning_rate", type=float, default=0.1) # 수정 필요
parser.add_argument("--g_norm", type=int, default=5)
parser.add_argument("--optim", type=str, default="sgd")
parser.add_argument("--L2s", type=int, default=5)
parser.add_argument("--use_earlystop", type=int, default=0)

parser.add_argument("--total_steps", type=int, default=50000) # 수정 필요
#parser.add_argument("--eval_period", type=int, default=400)
args = parser.parse_args()
config = load_config(args.config)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu")

lg = get_logger()
oj = os.path.join
print("Task : {}".format(args.task))
print("Dataset : {}".format(args.dataset))

assert args.task in ["sent", "tag"]
assert args.dataset in ["AG", "Sogou", "DBP", "Yelp_P", "Yelp_F", 'Yah_A', "Amz_F", "Amz_P"]

from src.model import FastText_Classifier as classifier
import src.train as train

# log record

task_loc = oj(args.log, args.task)
data_loc = oj(args.log, args.task, args.dataset)
tb_loc = oj(data_loc, 'tb')
chkpnt_loc = oj(data_loc, 'cnkpnt')

if not os.path.exists(task_loc):
    os.mkdir(task_loc)
    os.mkdir(data_loc)
    os.mkdir(tb_loc)
    os.mkdir(chkpnt_loc)

writer = SummaryWriter(tb_loc)

# data load
train_loader = train.get_data_loader(config, args.dataset, args.ngram,  "train")
dev_loader = train.get_data_loader(config, args.dataset, args.ngram, "dev")
test_loader = train.get_data_loader(config, args.dataset, args.ngram, "test")
print("Dataset iteration num : train {} | dev {} | test {}".format(len(train_loader),len(dev_loader) , len(test_loader)))

# model load
model = classifier(config=config, args=args)

# trainer load
trainer = train.get_trainer(config, args, train_loader, writer, type="train")
dev_trainer = train.get_trainer(config, args, dev_loader, writer, type='dev')
test_trainer = train.get_trainer(config, args, dev_loader, writer, type='test')


early_stop_loss = []
for epoch in range(1, args.epochs+1):
    trainer.train_epoch(model, epoch)
    valid_loss = dev_trainer.train_epoch(model, epoch, trainer.global_step)
    test_loss = test_trainer.train_epoch(model, epoch, trainer.global_step) ### 수정 필요
    early_stop_loss.append(valid_loss)

    if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]: 
        break

    #if not os.path.exists(cnkpnt_loc):
    #    os.mkdir(cnkpnt_loc)
    #torch.save({'epoch':epoch, 'model':model}, os.path.join(chkpnt_loc, "model.{}.ckpt".format(epoch)))

    train_accuracy = trainer.evaluator
    valid_accuracy = dev_trainer.evaluator
    test_accuracy = test_trainer.evaluator

    print('epoch : {} | train_accuracy : {} | valid_accuracy : {} | test_accuracy :{}'.format(epoch, train_accuracy, valid_accuracy, test_accuracy))

    
    









