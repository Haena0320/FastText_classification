import numpy as np
import copy

class FastText_Classifier:
    def __init__(self, config, args):
        data_info = config.data_info[args.dataset]

        if args.ngram ==1:
            self.vocab_size = data_info.vocab_size_1
        else:
            self.vocab_size = data_info.vocab_size_2
        self.batch_size = config.train.batch_size

        self.hidden = config.train.hidden
        self.n_classes = data_info.classes

        self.W_in = 0.1 * np.random.randn(self.vocab_size, self.hidden)
        self.W_in[0] = 0 # padding value == 0
        self.W_out = np.random.randn(self.hidden, self.n_classes)
        self.b = np.random.randn(self.n_classes)

    def forward(self, x, label): # (batch_size, sentence_max_len)
        self.t = label
        self.x = x
        self.sum_sentence_vec = np.mean(self.W_in[x], axis=1)       # (batch_size, sentence_max_len, hidden_size) -> (batch_size, hidden_size)
        output = np.dot(self.sum_sentence_vec, self.W_out) + self.b #(batch_size, n_classes)
        output = softmax(output)
        self.grad = copy.deepcopy(output)

        # accuracy
        y = np.argmax(output, axis=1) # (batch_size, )
        accuracy = len(y[y ==self.t])/len(y) * 100

        # loss
        loss = -np.sum(np.log(output[np.arange(len(self.t)), self.t]+1e-7))/len(self.t)
        return loss, accuracy

    def backward(self, learning_rate):
        self.grad[np.arange(len(self.t)), self.t] -= 1 # softmaxwithloss
        self.grad /= len(self.t)
        self.b -= learning_rate * np.sum(self.grad, axis=0)
        d_W_in_x = np.expand_dims(np.dot(self.grad, self.W_out.T), axis=1)
        self.W_out -= learning_rate * np.dot(self.sum_sentence_vec.T , self.grad)
        self.W_in[self.x] -= learning_rate * d_W_in_x
        self.W_in[0] = 0 # padding vector -> update xxx !
        return None

def softmax(x):
    max_ = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x-max_)
    #exp_x = np.exp(x)
    exp_x_sum = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x/exp_x_sum



