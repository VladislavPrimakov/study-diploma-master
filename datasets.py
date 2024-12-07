import numpy.random
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision import datasets
import torch, numpy as np


class BaseMNISTDataset(Dataset):
    def __init__(self, root, split, download=False):
        super().__init__()
        self.split = split
        self.transform = T.ToDtype(torch.float, scale=True)
        self.dataset = datasets.MNIST(root, train=(split == "train"), download=download)
        self.data = self.dataset.data.unsqueeze(1).clone()  # Добавляем канал (1 для серых изображений)
        np_arr = np.array(self.dataset.targets.clone())
        self.list_classes = np.unique(np_arr)
        self.grouped_examples = {i: np.where(np_arr == i)[0] for i in self.list_classes}

    def __len__(self):
        return int(len(self.dataset) / 2) if self.split == "train" else len(self.dataset)

    def __getitem__(self, index):
        image = self.transform(self.data[index].clone())
        return image, self.dataset[index][1]
        

class DatasetMNISTPair(BaseMNISTDataset):
    def __init__(self, root, split, download=False):
        super().__init__(root, split, download)

    def __getitem__(self, index):
        class1 = numpy.random.choice(self.list_classes)
        index1 = numpy.random.choice(self.grouped_examples[class1])
        image_1 = self.transform(self.data[index1].clone())
        # same class
        if index % 2 == 0:
            index2 = numpy.random.choice(self.grouped_examples[class1])
            while index2 == index1:
                index2 = numpy.random.choice(self.grouped_examples[class1])
            image_2 = self.transform(self.data[index2].clone())
            target = torch.tensor(1, dtype=torch.float)
        # different class
        else:
            class2 = numpy.random.choice(self.list_classes)
            while class2 == class1:
                class2 = numpy.random.choice(self.list_classes)
            index2 = numpy.random.choice(self.grouped_examples[class2])
            image_2 = self.transform(self.data[index2].clone())
            target = torch.tensor(0, dtype=torch.float)
        return image_1, image_2, target


class DatasetMNISTTriplet(BaseMNISTDataset):
    def __init__(self, root, split, download=False):
        super().__init__(root, split, download)

    def __getitem__(self, index):
        class1 = numpy.random.choice(self.list_classes)
        index1 = numpy.random.choice(self.grouped_examples[class1])
        anchor = self.transform(self.data[index1].clone())
        # same class
        index2 = numpy.random.choice(self.grouped_examples[class1])
        while index2 == index1:
            index2 = numpy.random.choice(self.grouped_examples[class1])
        positive = self.transform(self.data[index2].clone())
        # different class
        class2 = numpy.random.choice(self.list_classes)
        while class2 == class1:
            class2 = numpy.random.choice(self.list_classes)
        index3 = numpy.random.choice(self.grouped_examples[class2])
        negative = self.transform(self.data[index3].clone())
        return anchor, positive, negative


class DatasetMNISTConvolutional(BaseMNISTDataset):
    def __init__(self, root, split, download=False):
        super().__init__(root, split, download)

    def __getitem__(self, index):
        class1 = numpy.random.choice(self.list_classes)
        index1 = numpy.random.choice(self.grouped_examples[class1])
        image_1 = self.transform(self.data[index1].clone())
        # same class
        if index % 2 == 0:
            index2 = numpy.random.choice(self.grouped_examples[class1])
            while index2 == index1:
                index2 = numpy.random.choice(self.grouped_examples[class1])
            image_2 = self.transform(self.data[index2].clone())
            class2 = class1
        # different class
        else:
            class2 = numpy.random.choice(self.list_classes)
            while class2 == class1:
                class2 = numpy.random.choice(self.list_classes)
            index2 = numpy.random.choice(self.grouped_examples[class2])
            image_2 = self.transform(self.data[index2].clone())
        return image_1, image_2, class1, class2
