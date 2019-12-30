from torch.utils.data import Dataset
from abc import abstractmethod


class BaseIterator(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

        # batches is to store batches of data for torch
        self.batches = self.initialise()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    @abstractmethod
    def initialise(self):
        raise NotImplementedError
