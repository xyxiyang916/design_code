"""
构建设备类，包括终端，边缘服务器和云服务器
"""
import copy
import os

from torch import nn, optim

from train import *
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob
import nibabel as nib



class Device:
    def __init__(self, device_type, device_id, model):
        self.device_type = device_type
        self.device_id = device_id
        self.model = model.to(device)

class End(Device):
    global_model = None
    ood_test_result = 0
    cosine_test_result = 0
    flag = False
    pos = (0, 0)
    def __init__(self, device_type, device_id, model, train_loader, test_loader, ood_loader):
        Device.__init__(self, device_type, device_id, model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)  # 学习率衰减
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ood_loader = ood_loader

    def save_model(self, path, name):
        path = path + '/' + name + '.pth'
        print(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path, name):
        self.model.load_state_dict(torch.load(path + '/' + name + '.pth'))

    def train(self, epoch=1):
        for i in range(epoch):
            # 执行一次训练
            self.model.train().to(device)
            total_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            logger.debug(
                f"device:{self.device_type} | id:{self.device_id} | Epoch [{i+1}] | Train Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")
            self.scheduler.step()  # 更新学习率
            #train(i+1, self.model, self.train_loader, self.criterion, self.optimizer, self.scheduler)

    def test_id(self):
        self.model.eval().to(device)
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        logger.debug(
            f"device:{self.device_type} | id:{self.device_id} | Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")

    def test_ood(self):
        self.model.eval().to(device)
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.ood_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        logger.debug(
            f"device:{self.device_type} | id:{self.device_id} | Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")


    def ood_test(self, loop_num, id_loader, ood_loader):
        logger.debug(f'{self.device_id}号终端设备ood检测')
        # 先获取ID数据集的最短距离结果
        id_features, id_result = detect_ood_with_mahalanobis(self.model, id_loader, id_loader)
        # 再获取OOD数据集的最短距离结果
        ood_features, ood_result = detect_ood_with_mahalanobis(self.model, id_loader, ood_loader)
        threshold, B_ratio = evaluate_threshold(id_result, ood_result, 95)
        if B_ratio > 0.9:
            self.flag = True
            self.save_model(f'./ood/{loop_num}', f'id：{self.device_id}')
        logger.debug(f'阈值为：{threshold}，结果为：{B_ratio}')

    def cosine_test(self):
        logger.debug(f'{self.device_id}号终端设备余弦相似度检测')
        pass




class Edge(Device):
    def __init__(self, device_type, device_id, model, test_loader):
        Device.__init__(self, device_type, device_id, model)
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
    def model_aggregation(self, loop_num, end_model_list):
        if not len(end_model_list):
            return
        avg_state = {
            key: torch.zeros_like(param, dtype=torch.float32)
            for key, param in self.model.state_dict().items()
        }

        # 累加所有模型参数
        for end in end_model_list:
            model_state = end.model.state_dict()
            for key in avg_state:
                avg_state[key] += model_state[key]

        # 计算平均值
        for key in avg_state:
            avg_state[key] /= len(end_model_list)

        # 加载参数到新模型
        self.model.load_state_dict(avg_state)
        logger.debug(f'第{loop_num}次聚合完成')

    def model_test(self, loop_num):
        self.model.eval().to(device)
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        logger.debug(f"device:{self.device_type} | id:{self.device_id} | Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")


class Cloud(Device):
    def __init__(self, device_type, device_id, model):
        Device.__init__(self, device_type, device_id, model)

class Controller:
    def __init__(self, end_devices:list[End], edge_devices:list[Edge], cloud_device):
        self.end_devices = end_devices
        self.edge_devices = edge_devices
        self.cloud_device = cloud_device

    def model_train(self, loop_num, epoch):
        logger.debug(f'第{loop_num}次总训练')
        for end in self.end_devices:
            # 调用终端设备进行训练，默认训练一轮
            end.train(epoch)
            end.test_id()
            #end.test_ood()

    def model_evaluate(self, loop_num, id_loader, ood_loader):
        for end in self.end_devices:
            end.flag = False
        logger.debug(f'进行第{loop_num}次终端模型质量测试')
        for end in self.end_devices:
            # 进行ood和余弦检测，终端设备标记本轮是否参与聚合,双检测全部通过即可
            end.ood_test(loop_num, id_loader, ood_loader)
            #end.cosine_test()

    def model_cluster(self, loop_num):
        logger.debug(f'进行第{loop_num}次终端聚类')
        available_end = []
        for end in self.end_devices:
            if end.flag:
                available_end.append(end)
        logger.debug(f'可用终端模型数量为{len(available_end)}')
        for end in available_end:
            logger.debug(f'终端id：{end.device_id}')
        # 调用聚类函数
        # 边缘服务器进行模型聚合
        for edge in self.edge_devices:
            edge.model_aggregation(loop_num, available_end)
            edge.model_test(loop_num)
