import torch
from torch import flatten
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# data loader
def get_data_loader(config, dataset, n=1, type="train", shuffle=True, drop_last=True):
    batch_size = config.train.batch_size
    n_gram = str(n) + "_gram"
    data = Make_Dataset(config.path_prepro[dataset], n_gram, type)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, collate_fn=padded_seq)
    return data_loader


def padded_seq(samples):
    data = []
    label = []
    for sample in samples:
        data.append(sample["data"])
        label.append(sample["label"])
    data = padd(data)
    return data, label

def padd(samples):
    length = [len(s) for s in samples]
    max_length = max(length)
    batch = np.zeros((len(length), max_length), dtype=int)
    for idx, sample in enumerate(samples):
        batch[idx, :length[idx]] = sample
    return batch


class Make_Dataset(Dataset):
    def __init__(self, file_path, n_gram, type):
        data = torch.load(file_path)
        self.dataset = data[n_gram]

        if type == "train":
            self.data = self.dataset['train']
            self.data_label = self.dataset["train_label"]

        elif type == "dev":
            self.data = self.dataset["dev"]
            self.data_label = self.dataset["dev_label"]

        else:
            self.data = self.dataset["test"]
            self.data_label = self.dataset["test_label"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ret_dict = dict()
        ret_dict["data"] = self.data[idx]
        ret_dict['label'] = self.data_label[idx]

        return ret_dict


# training type 설정해서 -> Trainer 가져옴
def get_trainer(config, args, data_loader, writer, type="train"):
    return Trainer(config, args, data_loader, writer, type)

class Trainer:
    def __init__(self, config, args, data_loader, log_writer, type):
        self.config= config
        self.args = args
        
        self.data_loader = data_loader
        self.log_writer = log_writer
        self.type = type
        self.global_step = 0
        self.total_step = args.total_steps
        self.epochs = args.epochs
        self.max_learning_rate = config.train.max_learning_rate
        
    def lr_rate_scheduler(self):
        learning_rate = 1-self.global_step / (self.total_step * self.epochs)
        if learning_rate <= 0.001:
            learning_rate = 0.001
        else:
            learning_rate *= self.max_learning_rate
        return learning_rate

    
    def train_epoch(self, model, epoch, global_step=None):
        loss_save = list()
        for data in tqdm(self.data_loader, desc="Epoch : {}".format(epoch)):
            assert len(data[0])==len(data[1])
            loss, accuracy = model.forward(data[0], data[1])
            self.evaluator = accuracy
            if self.type =="train":
                learning_rate = self.lr_rate_scheduler()
                model.backward(learning_rate)
                self.global_step +=1
                self.write_tb(loss, global_step)
            else:
                loss_save.append(loss)

        if self.type != "train":
            loss = sum(loss_save)/len(loss_save)
            self.write_tb(loss, global_step)
            return loss

    def write_tb(self, loss, global_step):
        if self.type =="train":
            self.log_writer.add_scalar("train/loss", loss, global_step)
        else:
            self.log_writer.add_scalar("valid/loss", loss, global_step)







