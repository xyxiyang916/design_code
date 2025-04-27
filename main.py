import torch.nn as nn
import torch.optim as optim

from loguru import logger
from datasets import *
from models import *
from train import *

# 创建日志文件
logger.add("./log/log_{time}.log")
# 开始调试
logger.debug("开始调试")

global_loop = 10
num_edge = 1
num_end = 1


if __name__ == '__main__':
    logger.debug("开始调试主函数")
    logger.debug("-----创建数据集-----")
    # 加载数据集
    creat_datasets(num_end)
    # 全局初始化，生成每个终端设备和云服务器的初始模型
    logger.debug("-----创建初始化模型-----")
    creat_model(num_end, num_edge)
    # 训练阶段
    # for i in 总轮次
    # 终端设备到边缘服务器训练+聚合
    # for j in 每个终端设备
    # 对每一个终端设备进行本地模型训练
    # 设备聚类，绑定终端设备到边缘服务器
    # 边缘服务器到云服务器聚合
    logger.debug("-----开始全局训练轮次-----")
    logger.debug(f"总训练轮次：{global_loop}")
    for i in range(global_loop):
        logger.debug(f"模型训练轮次：{i+1}")
        for j in range(num_end):
            logger.debug(f"训练第{j + 1}号终端设备")
            # 训练代码
            # 读取模型
            #model = ResNet_18(BasicBlock, num_classes=10)
            model = resnet18(num_classes=10)
            model.load_state_dict(torch.load(f"models/end/end_model_{j + 1}.pth"))
            # 读取数据
            tra, tes = get_dataset()
            dataloader = DataLoader(tra, batch_size=128, shuffle=True)
            #dataloader = load_dataloader(f"fed_data/end_data_train_{j + 1}.pkl")
            # 执行训练
            print(f"Epoch [{i+1}] | ")
            train(
                model=model,
                dataloader=dataloader,
            )
            # 存储模型
            save_model(model, f"models/end/end_model_{j + 1}.pth")
        # 一轮训练完成
        logger.debug(f"模型聚合轮次：{i+1}")
        # 执行OOD检测，先得到模型给每个类别输出的特征向量



