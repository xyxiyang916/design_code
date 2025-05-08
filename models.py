import os

import torch
from torch import nn
from loguru import logger
import torch.nn.functional as F

# 性能更好的18版本
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

# 18、34版本残差块
'''
@:param
		input_channels: 输入通道数
		num_channels: 输出通道数
		use_1x1conv: 是否使用 1x1conv (使用就说明是卷积映射残差块, 要变换尺寸)
		strides: 步长
'''
# 残差快
class Residual_primary(nn.Module):
	def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
		# 默认 use_1x1conv=False, strides=1
		# 继承父类所有方法
		super(Residual_primary, self).__init__()
		# 第一个卷积块，卷积核大小3*3，遍历图像的步长1
		self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
		# 归一化
		self.bn1 = nn.BatchNorm2d(num_channels)  # BN层
		# ReLU激活
		self.relu1 = nn.ReLU(inplace=True)  # Inplace ReLU to save memory
		# 第二个卷积核，接前一个卷积核输出
		self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
		# 归一化
		self.bn2 = nn.BatchNorm2d(num_channels)
		# 如果用了 1x1conv 就说明是卷积映射残差块, 要变换尺寸，1*1卷积核，且步长为2，大小减半
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
			self.bn3 = nn.BatchNorm2d(num_channels)
		else:
			self.conv3 = None

	def forward(self, X):
		Y = self.conv1(X)
		Y = self.bn1(Y)
		Y = self.relu1(Y)
		Y = self.conv2(Y)
		Y = self.bn2(Y)
		if self.conv3:
			# 第一个卷积核的输入为56*56*64，经过残差块后不变，可以直接相加
			# 第二个残差快开始，经过残差块后，图像尺寸减半，通道翻倍，要对原始输入进行变换才能和残差结果相加
			X = self.conv3(X)
			X = self.bn3(X)  # If using 1x1 conv, also apply BN
		# 加上输入再ReLU
		Y += X
		Y = nn.ReLU(inplace=True)(Y)  # Optionally, you can move this ReLU outside the Residual class
		return Y


# 大残差结构
'''
@:param
		input_channels: 输入通道数
		num_channels: 输出通道数
		num_residuals: 残差块的个数
		first_block: 是否是第一个大残差结构
@:return
		nn.Sequential(*blk): 大残差结构
'''
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
	blk = []
	for i in range(num_residuals):
		# 第一个块时步长全为1，后面的残差块中第一个卷积核步长为2，步长为2会尺寸减半，第一个残差块接收56*56后不减半
		stride = 2 if i == 0 and not first_block else 1  # 从第二个大残差结构开始, 结构中的第一个残差块一般都会尺寸减半, 即 stride=2
		# 非第一个块且第一个卷积核
		use_1x1conv = i == 0 and not first_block  # use_1x1conv = False/True, 从第二个大残差结构开始, 结构中的第一个残差块都是卷积映射残差块
		if i == 0:
			blk.append(Residual_primary(in_channels, out_channels, use_1x1conv=use_1x1conv, strides=stride))
		else:
			blk.append(Residual_primary(out_channels, out_channels, strides=stride))
	# 将blk列表中所有元素以一个一个参数的形式返回
	return nn.Sequential(*blk)


# ResNet-18
def resnet18(num_classes, in_channels=3):
	# net = nn.Sequential(b1)
	net = nn.Sequential(
		# 第一个卷积，7*7，输出维度64，步长为2，将224*224的图像变为112*112*64，一共64个卷积核，每个卷积核三通道，(输入+2*padding-kernel)/stride下取整再加1
		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
		# 归一化
		nn.BatchNorm2d(64),
		# ReLU激活
		nn.ReLU(inplace=True),
		# maxPool后得到56*56*64的输出
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
	)
	# 加入残差快，第一个块和输入不等于输出时会进行维度变化
	# 第一个块，64个卷积核，输入通道和和本层通道相同
	net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
	net.add_module("resnet_block2", resnet_block(64, 128, 2))
	net.add_module("resnet_block3", resnet_block(128, 256, 2))
	net.add_module("resnet_block4", resnet_block(256, 512, 2))
	# 平均池化,1*1,不改变维度
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
	# 展平为一列，然后全连接
	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
	# Optionally, initialize weights here (e.g., nn.init.kaiming_normal_(net[0].conv1.weight, mode='fan_out', nonlinearity='relu'))
	return net


# ResNet-34
def resnet34(num_classes, in_channels=3):
	# 也可以使用语句 net = nn.Sequential(b1) 来代替下方
	net = nn.Sequential(
		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
	)
	net.add_module("resnet_block1", resnet_block(64, 64, 3, first_block=True))
	net.add_module("resnet_block2", resnet_block(64, 128, 4))
	net.add_module("resnet_block3", resnet_block(128, 256, 6))
	net.add_module("resnet_block4", resnet_block(256, 512, 3))
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
	# Optionally, initialize weights here
	return net

# 50版本残差块
class Bottleneck(nn.Module):
	def __init__(self, in_channels, out_channels, strides=1, downsamples=False):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels // 4)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=strides, padding=1,
		                       bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels // 4)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels)
		if downsamples:
			self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
			self.bn4 = nn.BatchNorm2d(out_channels)
		else:
			self.conv4 = None

	def forward(self, x):
		identity = x  # 映射

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.conv4:
			identity = self.conv4(identity)
			identity = self.bn4(identity)

		out += identity
		out = nn.ReLU(inplace=True)(out)
		return out


# 大残差结构
def resnet_Bottleneck(in_channels, out_channels, num_residuals, first_block=False):
	blk = []
	for i in range(num_residuals):
		stride = 2 if i == 0 and not first_block else 1  # 第一个残差结构中的第一个残差块，只改变通道数，不作下采样
		downsample = i == 0  # downsample = False/True, 从第二个大残差结构开始, 结构中的第一个残差块都是下采样映射残差块
		blk.append(Bottleneck(in_channels, out_channels, strides=stride, downsamples=downsample))
		in_channels = out_channels
	return nn.Sequential(*blk)


# ResNet-50
def resnet50(num_classes, in_channels=3):
	net = nn.Sequential(
		nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
		nn.BatchNorm2d(64),
		nn.ReLU(inplace=True),
		nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

	net.add_module("layer1", resnet_Bottleneck(64, 256, 3, first_block=True))
	net.add_module("layer2", resnet_Bottleneck(256, 512, 4))
	net.add_module("layer3", resnet_Bottleneck(512, 1024, 6))
	net.add_module("layer4", resnet_Bottleneck(1024, 2048, 3))
	net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
	net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(2048, num_classes)))

	return net

def creat_model(num_end, num_edge):
	logger.debug("创建终端设备初始化模型")
	for i in range(num_end):
		if not os.path.exists(f"models/end/end_model_{i + 1}.pth"):
			model = resnet18(num_classes=10)
			#model = ResNet_18(BasicBlock, num_classes=10)
			save_model(model, f"models/end/end_model_{i + 1}.pth")
	logger.debug("创建边缘服务器初始化模型")
	for i in range(num_edge):
		if not os.path.exists(f"models/edge/edge_model_{i + 1}.pth"):
			model = resnet34(num_classes=10)
			save_model(model, f"models/edge/edge_model_{i + 1}.pth")
	logger.debug("创建云服务器初始化模型")
	if not os.path.exists(f"models/cloud/cloud_model.pth"):
		model = resnet50(num_classes=10)
		save_model(model, f"models/cloud/cloud_model.pth")

def _save_model(model, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(model.state_dict(), path)