import torch
import torchvision

import transform

def build_dataset(dataset, batch_size_train, batch_size_test, num_workers=8, path='./datasets'):
    if dataset == "CIFAR10":
        num_class = 10
        mu = [0.4914, 0,4822, 0,4465]
        std = [0.2023, 0.1994, 0.2010]
        train_dataset_1 = torchvision.datasets.CIFAR10(
            root=path,
            download=True,
            train=True,
            transform=transform.Transforms(size=32, s=0.5),
        )
        test_dataset_1 = torchvision.datasets.CIFAR10(
            root=path,
            download=True,
            train=False,
            transform=transform.Transforms(size=32, s=0.5),
        )
        dataset_for_train = torch.utils.data.ConcatDataset([train_dataset_1, test_dataset_1])
        
        train_dataset_2 = torchvision.datasets.CIFAR10(
            root=path,
            download=True,
            train=True,
            transform=transform.Transforms(size=32).test_transform,
        )
        test_dataset_2 = torchvision.datasets.CIFAR10(
            root=path,
            download=True,
            train=False,
            transform=transform.Transforms(size=32).test_transform,
        )
        dataset_for_test = torch.utils.data.ConcatDataset([train_dataset_2, test_dataset_2])
    else:
        raise NotImplementedError
        
    data_loader = torch.utils.data.DataLoader(
        dataset_for_train,
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_for_test,
        batch_size=batch_size_test,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader, data_loader_test, num_class