import torch
from torch.utils.data import Dataset
import numpy as np
import copy
from abc import abstractmethod


class BaseIterator(Dataset):
    def __init__(self, file_path, batch_size, device=torch.device('cpu')):
        self.file_path = file_path
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return len(self.batches)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def initialise_batches(self):
        pass


class ReferentialBaseIterator(BaseIterator):
    """
    A base iterator for referential games, futher illustrations are:
    1. every batch is a dictionary containing 2 keys: 'data' and 'label', e.g. {'data': img_matrix, 'label': caption}
    2. every element is a dictionary containing 3 keys: 'correct', 'candidates' and 'label'.
    """
    def __init__(self, file_path, batch_size, num_distractors=14, device=torch.device('cpu')):
        super().__init__(file_path, batch_size, device=device)
        self.num_distractors = 14
        
        self.batches = None
        self.batch_indices = None

    @abstractmethod
    def initialise_batches(self):
        pass

    def __getitem__(self, idx):
        correct_batch = self.batches[idx]

        candidate_batches = [
            self._generate_distracting_batch(idx) for _ in range(self.num_distractors+1)
        ]

        golden_idx = np.random.randint(0, high=self.num_distractors+1, size=(correct_batch['data'].shape[0]))

        for i in range(correct_batch['data'].shape[0]):
            candidate_batches[golden_idx[i]]['data'][i, :, :, :] = correct_batch['data'][i, :, :, :]
            candidate_batches[golden_idx[i]]['label'][i] = correct_batch['label'][i]

        golden_idx = torch.from_numpy(golden_idx).to(self.device).to(torch.long)

        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def _generate_distracting_batch(self, idx):
        sample_idx = np.random.choice(self.batch_indices)
        while self.batch_size == 1 and sample_idx == idx:
            sample_idx = np.random.choice(self.batch_indices)

        if sample_idx == idx:
            return self._reperm_batch(idx)
        else:
            return copy.deepcopy(self.databatch_set[sample_idx])

    def _reperm_batch(self, idx):
        # the "batch size" is not necessarily self.batch_size as the last batch may contain fewer samples than others.
        _batch_size = self.batches[idx]['data'].shape[0]

        original_idx = torch.arange(_batch_size, device=self.device)
        new_idx = torch.randperm(_batch_size, device=self.device)

        while not (original_idx == new_idx).sum().eq(0):
            new_idx = torch.randperm(_batch_size, device=self.device)

        shuffled_imgs = self.batches[idx]['data'][new_idx]
        shuffled_labels = [self.batches[idx]['label'][i] for i in new_idx]

        return {
            'data': shuffled_imgs,
            'label': shuffled_labels,
        }

