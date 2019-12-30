from .BaseIterator import BaseIterator


class ImageIterator(BaseIterator):
    def __init__(self, file_path, batch_size, num_distractors=14, device=torch.device('cpu')):
        super().__init__(file_path, batch_size, device=device)
        self.num_distractors = num_distractors

        self.batches = self.initialise()

    def __getitem__(self, idx):
        correct_batch = self.databatch_set[idx]

        candidate_batches = [
            self.generate_distractor_batch(idx) for _ in range(self.num_distractors+1)
        ]

        golden_idx = np.random.randint(0, high=self.num_distractors+1, size=(correct_batch['imgs'].shape[0]))

        for i in range(correct_batch['imgs'].shape[0]):
            candidate_batches[golden_idx[i]]['imgs'][i, :, :, :] = correct_batch['imgs'][i, :, :, :]
            candidate_batches[golden_idx[i]]['label'][i] = correct_batch['label'][i]

        golden_idx = torch.from_numpy(golden_idx).to(self.device).to(torch.long)

        return {
            'correct': correct_batch,
            'candidates': candidate_batches,
            'label': golden_idx
        }

    def initialise(self):
        raise NotImplementedError

