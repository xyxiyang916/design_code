import csv
import os

import dill
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from loguru import logger
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToPILImage

torch.manual_seed(0)


def get_transform():
    return [  # 预处理训练集数据函数
        transforms.Compose([
            # 对32*32图像进行4像素0填充，再裁剪区域
            transforms.RandomCrop(32, padding=4),
            # 空间对称增强
            transforms.RandomHorizontalFlip(),
            # 转为张量
            transforms.ToTensor(),
            # 图像色彩标准化
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),

        # 预处理测试集数据函数
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),

        # 预处理测试集数据函数
        transforms.Compose([
            transforms.ToTensor(),
        ])
    ]

# test
def get_dataset(transform_mode):
    # 加载训练集和测试集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=get_transform()[transform_mode]
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=get_transform()[transform_mode]
    )
    return train_dataset, test_dataset


def creat_datasets(
        num_clients: int = 10,
        alpha: float = 0.5,  # non-iid程度的超参数，我喜欢用0.5和0.3
        flag: bool = False
):
    # 检查是否已经存在数据集
    for i in range(num_clients):
        if not os.path.exists(f"fed_data/end_data_train_{i + 1}.pth"):
            flag = True
    if not flag:
        logger.debug("数据集已存在")
        return

    # 读取本地数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transform()[0])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transform()[1])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    client_indices = dirichlet_distribution_noniid(train_dataset, num_clients, alpha)

    # 创建客户端数据加载器
    train_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=32, shuffle=True) for indices in
                     client_indices]

    return train_loaders, test_loader

    '''# 存储训练集
    for i, train_loader in enumerate(train_loaders):
        # 保存 DataLoader
        save_dataloader(train_loader, f'fed_data/end_data_train_{i + 1}.pkl')
    # 存储测试集
    save_dataloader(test_loader, 'fed_data/end_data_test.pkl')'''


def dirichlet_distribution_noniid(dataset, num_clients, alpha):
    # 获取每个类的索引
    class_indices = [[] for _ in range(10)]
    for idx, (image, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 使用Dirichlet分布进行数据划分
    client_indices = [[] for _ in range(num_clients)]
    for class_idx in class_indices:
        np.random.shuffle(class_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        client_split = np.split(class_idx, proportions)
        for client_idx, client_split_indices in enumerate(client_split):
            client_indices[client_idx].extend(client_split_indices)

    return client_indices


def split_by_class(dataset, classes_num = 10):
    """将数据集按类别拆分为子集字典"""
    class_indices = {i: [] for i in range(classes_num)}  # 创建10个空列表
    # 遍历数据集收集索引
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    # 创建子集字典
    class_datasets = {
        i: Subset(dataset, indices)
        for i, indices in class_indices.items()
    }
    return class_datasets


def create_class_loaders(class_datasets, batch_size=32, shuffle=True):
    """为每个类别创建独立DataLoader"""
    class_loaders = {}
    for class_idx, subset in class_datasets.items():
        class_loaders[class_idx] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    return class_loaders


def save_dataloader(loader, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        dill.dump(loader, f)


def load_dataloader(path):
    with open(path, 'rb') as f:
        return dill.load(f)


to_pil = ToPILImage()


def save_cifar10_images_labels(dataset, root_dir='./cifar10_data', image_format='png'):
    """
    将CIFAR-10数据集保存为图片文件+CSV标签文件

    参数：
        root_dir (str): 保存根目录路径
        image_format (str): 图片保存格式（png/jpeg等）
    """
    # 定义数据集类别名称
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # 创建目录结构
    dirs = {
        'train_images': os.path.join(root_dir, 'images/train'),
        'test_images': os.path.join(root_dir, 'images/test'),
        'labels': os.path.join(root_dir, 'labels')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 定义保存函数
    def _save_subset(dataset, mode='train'):
        csv_path = os.path.join(dirs['labels'], f'{mode}_labels.csv')
        image_dir = dirs[f'{mode}_images']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label', 'label_name'])
            for idx, (image_tensor, label) in enumerate(dataset):
                # 转换张量为PIL图像
                image = to_pil(image_tensor)
                # 生成文件名
                filename = f"{mode}_{idx:05d}.{image_format}"
                filepath = os.path.join(image_dir, filename)

                # 保存图片
                if image_format.lower() == 'jpeg':
                    image.save(filepath, quality=95)
                else:
                    image.save(filepath)

                # 写入CSV
                writer.writerow([filename, label, classes[label]])

    _save_subset(dataset, 'train')
    print(f"数据保存完成！目录结构：\n{root_dir}")
    print(f"图片路径：{dirs['train_images']} 和 {dirs['test_images']}")
    print(f"标签文件：{dirs['labels']}/train_labels.csv 和 test_labels.csv")

'''去除排除类别后的cifar100数据集'''
class CIFAR100OODWrapper(CIFAR100):
    def __init__(self, exclude_classes, **kwargs):
        super().__init__(**kwargs)

        # 获取CIFAR-100类别到索引的映射
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        # 筛选有效OOD类别
        valid_indices = [
            i for i, name in enumerate(self.classes)
            if name not in exclude_classes
        ]

        # 重构数据
        self.data = [img for i, img in enumerate(self.data) if self.targets[i] in valid_indices]
        self.targets = [target for target in self.targets if target in valid_indices]

'''创建基于cifar-100的OOD数据集'''
def get_ood_cifar100(transform_mode):
    # CIFAR-10类别列表
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    # 需要排除的CIFAR-100细分类别（示例）
    excluded_cifar100_classes = [
        'baby', 'boy', 'girl', 'man', 'woman',  # 人类类别明显不同
        'beaver', 'dolphin', 'otter', 'seal',  # 水生动物
        'maple_tree', 'oak_tree', 'palm_tree',  # 植物类
        'mountain', 'forest',  # 自然场景
        'plate', 'bowl', 'bottle'  # 日常物品
    ]
    return CIFAR100OODWrapper(
        root='./data',
        train=False,  # 使用测试集作为OOD数据
        download=True,
        transform=get_transform()[transform_mode],
        exclude_classes=excluded_cifar100_classes
    )
