import random

import torch.nn as nn
import torch.optim as optim

from loguru import logger
from datasets import *
from models import *
from devices import *

seed_value = 2025  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True

# 创建日志文件
logger.add("./log/log_{time}.log")
# 开始调试
logger.debug("开始调试")

split = 3
global_loop = 20
num_edge = 1
num_end = 20
epoch = 5


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型（注意传入 BasicBlock）
    '''for epoch in range(1, 100 + 1):
        train(epoch, model, tra_loader, criterion, optimizer, scheduler)
        if epoch % 10 == 0:  # 每 10 个 epoch 测试一次
            test(epoch, model, tes_loader, criterion)'''

    logger.debug("开始调试主函数")
    logger.debug("进行初始化")
    logger.debug("初始化数据集")
    # 训练数据集，测试数据集
    train_loader_list, test_loader = creat_datasets(split)
    # id数据集
    id_class_loader = create_class_loaders(split_by_class(test_loader.dataset, 10))
    # ood数据集
    ood_data = get_ood_cifar100(1)
    ood_loader = DataLoader(ood_data, batch_size=32, shuffle=False)
    ood_class_datasets = split_by_class(ood_data, 100)
    ood_class_loaders = create_class_loaders(ood_class_datasets, shuffle=False)
    logger.debug("初始化模型")
    logger.debug("初始化设备")
    end_devices_list = []
    edge_devices_list = []
    for i in range(num_end):
        end_devices_list.append(End('end', i + 1, ResNet_18(BasicBlock, num_classes=10), train_loader_list[random.randint(0,split-1)], test_loader, ood_loader))
    for i in range(num_edge):
        edge_devices_list.append(Edge('edge', i + 1, ResNet_18(BasicBlock, num_classes=10), test_loader))
    controller = Controller(end_devices_list, edge_devices_list, Cloud('cloud', 0, ResNet_18(BasicBlock, num_classes=10)))
    logger.debug("开始训练")
    logger.debug(f"总训练轮次：{global_loop}")
    for i in range(global_loop):
        controller.model_train(i+1, epoch)
        controller.model_evaluate(i+1, id_class_loader, ood_class_loaders)
        # 一轮训练完成

        logger.debug(f"第{i+1}轮模型聚合")
        #controller.model_cluster(i+1)
        # 执行OOD检测，先得到模型给每个类别输出的特征向量



