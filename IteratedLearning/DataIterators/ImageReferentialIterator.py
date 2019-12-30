from .BaseIterator import ReferentialBaseIterator

import os, math
import torch, torchvision


class ImageReferentialIterator(ReferentialBaseIterator):
    def __init__(self, file_path, batch_size, num_distractors=14, device=torch.device('cpu')):
        super().__init__(file_path, batch_size, num_distractors=num_distractors, device=device)
        self.batches = self.initialise_batches()

    @staticmethod
    def _load_img_set(self, dir_path):
        img_file_names = os.listdir(dir_path)
        imgs = [Image.open(os.path.join(dir_path, name)).convert('RGB') for name in img_file_names]
        return img_file_names, imgs

    @staticmethod
    def _build_img_tensors(imgs, device=args.device):
        tensors = []
        for img in imgs:
            tensors.append(torchvision.transforms.ToTensor()(img))
        tensors = torch.stack(tensors).to(device)
        return tensors

    def initialise_batches(self):
        batches = []

        img_names, imgs = self.load_img_set(self.file_path)
        img_names = [name.split('.')[0] for name in img_names]

        assert len(img_names) == len(imgs)
        c = list(zip(img_names, imgs))
        img_names, imgs = zip(*c)
        img_names = list(img_names)
        imgs = list(imgs)

        # ceil/floor
        if len(imgs) < self.batch_size:
            num_batches = 1
        else:
            num_batches = math.floor(len(imgs) / self.batch_size)

        for i in range(num_batches):
            img_batch = self._build_img_tensors(
                    imgs[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))],
                    device=self.device
                )
            img_label_batch = img_names[i*self.batch_size: min((i+1)*self.batch_size, len(imgs))]

            batches.append({
                'imgs': img_batch,
                'label': img_label_batch,
            })

        return batches

