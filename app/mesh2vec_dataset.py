import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn

class MeSH2VecDataset(Dataset):


    def __init__(self, f_input, f_target, f_vocab):
        self.input = self.read(f_input).tolist()
        self.target = self.read(f_target).squeeze().tolist()
        self.vocab = self.read(f_vocab).squeeze().tolist()
        self.vocab = dict(zip(self.vocab, range(len(self.vocab))))
        self.n = len(self.input)
        self.w = len(self.input[0])

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        x = torch.tensor([self.vocab[m] for m in self.input[index]], dtype=torch.int32)
        y = torch.tensor(self.vocab[self.target[index]], dtype=torch.int32)
        return x, y

    def read(self, path):
        input = pd.read_csv(path, header=None, dtype=str)
        return input.values
