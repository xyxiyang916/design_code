import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

flag = [1, 1, 1, 1]

seed_value = 2025  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

'''-------------数据加载（复用你的代码）-----------------------------'''
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

'''-------------模型定义（复用你的代码）-----------------------------'''


class BasicBlock(nn.Module):
	def __init__(self, inchannel, outchannel, stride=1):
		super(BasicBlock, self).__init__()
		self.left = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(outchannel),
			nn.ReLU(inplace=True),
			nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(outchannel)
		)
		self.shortcut = nn.Sequential()
		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(outchannel)
			)

	def forward(self, x):
		out = self.left(x)
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet_18(nn.Module):
	def __init__(self, ResidualBlock, num_classes=10):
		super(ResNet_18, self).__init__()
		self.inchannel = 64
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),
		)
		self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
		self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
		self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
		self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
		self.fc = nn.Linear(512, num_classes)

	def make_layer(self, block, channels, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.inchannel, channels, stride))
			self.inchannel = channels
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out


'''-------------训练配置-----------------------------'''
def save_model(path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(model.state_dict(), path)

# 初始化模型（注意传入 BasicBlock）
model = ResNet_18(BasicBlock, num_classes=10).to(device)
print('存储初始化模型')
save_model('./0%正确率模型.pth')
#model.load_state_dict(torch.load(f"./end_model_1.pth"))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率衰减

'''-------------训练函数-----------------------------'''


def train(epoch):
	model.train()
	total_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

	print(f"Epoch [{epoch}] | Train Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")
	scheduler.step()  # 更新学习率


'''-------------测试函数-----------------------------'''
flag = [0]
def test():
	model.eval()
	total_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			total_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	if correct / total > 0.9:
		print('存储90%正确率模型')
		save_model(f'./90%正确率模型_{flag[0]}.pth')
		flag[0]+=1
	print(f"Test Results | Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")

'''-------------开始训练-----------------------------'''

for epoch in range(1, 100 + 1):
	train(epoch)
	if flag[0] > 5:
		exit(0)
	if epoch > 80:  # 每 10 个 epoch 测试一次
		test()

# 保存模型
torch.save(model.state_dict(), '95%正确率模型.pth')
print("训练完成，模型已保存！")