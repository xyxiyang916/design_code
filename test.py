import random

import pandas as pd
import torch

from datasets import *
from models import *
from train import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob
import nibabel as nib
seed_value = 2025  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True

# 原始图像不同类别差异性分析
#test_pic_diff()

# TSNE图绘制
model1 = ResNet_18(BasicBlock, num_classes=10)
model2 = ResNet_18(BasicBlock, num_classes=10)
model3 = ResNet_18(BasicBlock, num_classes=10)
model4 = ResNet_18(BasicBlock, num_classes=10)

model_list = []
model1.load_state_dict(torch.load('./90%正确率模型_0.pth'))
model2.load_state_dict(torch.load('./90%正确率模型_1.pth'))
model3.load_state_dict(torch.load('./90%正确率模型_2.pth'))
model4.load_state_dict(torch.load('./90%正确率模型_3.pth'))



model_list.append(model1)
model_list.append(model2)
model_list.append(model3)
model_list.append(model4)

avg_model = ResNet_18(BasicBlock, num_classes=10)
avg_state = {
    key: torch.zeros_like(param, dtype=torch.float32)  # ✅ 强制为浮点类型
    for key, param in avg_model.state_dict().items()
}

# 累加所有模型参数
for model in model_list:
    model_state = model.state_dict()
    for key in avg_state:
        avg_state[key] += model_state[key]

# 计算平均值
for key in avg_state:
    avg_state[key] /= len(model_list)

# 加载参数到新模型
avg_model.load_state_dict(avg_state)

criterion = nn.CrossEntropyLoss()
train_loader_list, test_loader = creat_datasets(1)
test(1, avg_model, test_loader, criterion)

exit(0)



