import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn

class MeSH2VecDataset(Dataset):
    """
    Create a Dataset for MeSH2VecDataset

    Args:
        f_input (str): path to the input file. A comma separated csv file where each line correspond to a context
        f_target (str): path to the target file. A csv file where each line correspond to target MeSH. Must be the same number of lines as in the context file.
        f_vocab (str): path to the vocabulary file. A csv file, one line by MeSH. The line number will five the id of the MeSH in the vocab.
        f_labels (str): path to the labels file. A two columns csv file, with MeSH identifier in the first column and the corresponding label in the second 

    Return: a list of two tensors: (tensor of length w corresponding to the context input ids, tensor of length 1 corresponding to the id of the target MeSH)
    """


    def __init__(self, f_input, f_target, f_vocab, f_labels):
        self.input = self.read(f_input).tolist()
        self.target = self.read(f_target).squeeze().tolist()
        self.vocab = self.read_vocab(f_vocab)
        self.labels = self.mesh2label_dict(f_labels)
        self.n = len(self.input)
        self.w = len(self.input[0])

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        x = torch.tensor([self.vocab[m] for m in self.input[index]], dtype=torch.int32)
        y = torch.tensor(self.vocab[self.target[index]], dtype=torch.int32)
        return x, y

    def read(self, path):
        f = pd.read_csv(path, header=None, dtype=str)
        return f.values

    def read_vocab(self, f_vocab):
        v = self.read(f_vocab).squeeze().tolist()
        return dict(zip(v, range(len(v))))

    def mesh2label_dict(self, f_labels):
        labels = self.read(f_labels)
        return dict(zip(labels[:,0].tolist(), labels[:,1].tolist()))