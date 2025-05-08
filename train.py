import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, model, loader, criterion, optimizer, scheduler):
	model.train().to(device)
	total_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(loader):
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

	logger.debug(f"Epoch [{epoch}] | Train Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")
	scheduler.step()  # 更新学习率

def test(epoch, model, loader, criterion):
	model.eval().to(device)
	total_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			total_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	print(f"Test Results | Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")


def extract_features(model, dataloader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	features = []
	with torch.no_grad():
		# 每次输入一个batch
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			inputs = inputs.to(device)
			feats = model(inputs)
			feats = feats.view(feats.size(0), -1)  # Flatten if needed
			features.append(feats.cpu())
	# 返回一个类别所有样本的特征向量
	return torch.cat(features, dim=0)  # Shape: [N_samples, feature_dim]

'''执行带有马氏距离的OOD检测，传入模型，ID数据集，OOD数据集'''
def detect_ood_with_mahalanobis(model, id_dataloaders: dict, ood_dataloader: dict):
	id_class_means = []
	all_id_features = []

	# 获取ID数据集的特征，是一个类别所有数据的特征张量
	for label, loader in id_dataloaders.items():
		feats = extract_features(model, loader)
		# 计算平均张量
		class_mean = feats.mean(dim=0)  # [feature_dim]
		# 加入类别平均特征列表
		id_class_means.append(class_mean)
		# 类别数据的特征原始数据
		all_id_features.append(feats)

	# 将每个类的平均特征的张量合为一个张量，类型数量*特征长度
	id_class_means = torch.stack(id_class_means, dim=0)  # [num_classes, feature_dim]
	# 所有的ID数据，每个类别的每个样本特征都合在一个张量中
	all_id_features = torch.cat(all_id_features, dim=0)  # [total_id_samples, feature_dim]

	# 对转置结果计算协方差，对所有样本计算
	cov = torch.cov(all_id_features.T)  # [feature_dim, feature_dim]
	# 向协方差矩阵的对角线添加一个微小值（1e-5），以提高数值稳定性
	cov += 1e-5 * torch.eye(cov.shape[0])  # For numerical stability
	# 计算伪逆矩阵
	inv_cov = torch.linalg.pinv(cov)  # [feature_dim, feature_dim]

	# 计算马氏距离
	ood_scores = []
	all_test_features = []
	all_test_features_tensor = None
	for label, loader in ood_dataloader.items():
		if len(loader.dataset) == 0:
			continue
		# 获取一个类别的特征列表
		feats = extract_features(model, loader)
		all_test_features.append(feats)
		# 记录一个类别中所有样本的最短距离
		class_dists = []
		for feat in feats:
			# 每个batch中每一个样本的结果
			# 记录每个样本相对每个类别的距离
			dists = []
			# 遍历每个类别的平均特征
			for mu in id_class_means:
				# 当前样本和类别平均特征差值
				diff = (feat.cpu() - mu)  # Move to cpu for matmul
				# 距离计算
				dist = diff @ inv_cov @ diff.T
				# 存储该样本相对于每个平均特征的距离值
				dists.append(dist.item())
			# 找出一个类别中一个样本距离类别平均特征最短的距离
			class_dists.append(min(dists))  # Take minimum distance to any class center
		# 记录每个类别中样本最短距离为类别OOD得分
		ood_scores.append(min(class_dists))
		all_test_features_tensor = torch.cat(all_test_features, dim=0)
	return all_test_features_tensor, ood_scores

import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

def extract_visual_statistics(dataset):
	"""
	输入：dataset 是一个 PyTorch Dataset，返回颜色、亮度、纹理特征的统计指标字典。
	"""
	rgb_means, rgb_stds = [], []
	brightness_vals = []
	texture_features = []

	for img, _ in dataset:
		if isinstance(img, torch.Tensor):
			img = img.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C

		# RGB通道均值和标准差
		rgb_means.append(np.mean(img, axis=(0, 1)))
		rgb_stds.append(np.std(img, axis=(0, 1)))

		# 亮度（灰度）分析
		gray = rgb2gray(img)
		brightness_vals.append(np.mean(gray))

		# 纹理（使用GLCM）
		gray_uint8 = (gray * 255).astype(np.uint8)
		glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
		contrast = graycoprops(glcm, 'contrast')[0, 0]
		energy = graycoprops(glcm, 'energy')[0, 0]
		homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
		entropy = -np.sum(glcm * np.log2(glcm + 1e-8))  # 自定义熵

		texture_features.append([contrast, energy, homogeneity, entropy])

	return {
		'rgb_mean': np.mean(rgb_means, axis=0),
		'rgb_std': np.mean(rgb_stds, axis=0),
		'brightness_mean': np.mean(brightness_vals),
		'brightness_std': np.std(brightness_vals),
		'texture_mean': np.mean(texture_features, axis=0),
		'texture_std': np.std(texture_features, axis=0),
	}

import matplotlib.pyplot as plt

def plot_radar_chart(feature_dicts, class_names, title='Radar Chart'):
	"""
	feature_dicts: List[Dict] - 每个类别提取后的统计特征
	class_names: List[str] - 对应类别的名称
	"""
	# 把所有的特征拼成统一的向量：mean_rgb(3) + std_rgb(3) + brightness(2) + texture(4) = 12维
	vectors = []
	for stats in feature_dicts:
		vec = np.concatenate([
			stats['rgb_mean'], stats['rgb_std'],
			[stats['brightness_mean'], stats['brightness_std']],
			#stats['texture_mean']
		])
		vectors.append(vec)

	print(vectors)
	vectors = np.array(vectors)

	# Min-Max 归一化
	scaler = MinMaxScaler()
	normalized = scaler.fit_transform(vectors)

	# 雷达图角度设置
	num_vars = vectors.shape[1]
	angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
	angles += angles[:1]  # 闭合

	# 特征标签
	labels = ['RGB_Mean_R', 'RGB_Mean_G', 'RGB_Mean_B',
			  'RGB_Std_R', 'RGB_Std_G', 'RGB_Std_B',
			  'Brightness_Mean', 'Brightness_Std',
			  #'Texture_Contrast', 'Texture_Energy',
			  #'Texture_Homogeneity', 'Texture_Entropy'
			  ]
	labels += labels[:1]

	# 绘图
	fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
	for i, (row, name) in enumerate(zip(vectors, class_names)):
		row = row.tolist()
		row += row[:1]  # 闭合
		ax.plot(angles, row, label=name, linewidth=2)
		ax.fill(angles, row, alpha=0.1)

	ax.set_title(title, fontsize=16)
	ax.set_theta_offset(np.pi / 2)
	ax.set_theta_direction(-1)
	ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])  # 修复：去掉多余标签
	ax.set_ylim(0, 1)
	ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
	plt.tight_layout()
	plt.show()


from scipy.spatial.distance import euclidean, cosine
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
import ot  # POT for Wasserstein distance
from torch.utils.data import DataLoader
def compute_js_divergence(p, q, eps=1e-8):
	p = p / (p.sum() + eps)
	q = q / (q.sum() + eps)
	m = 0.5 * (p + q)
	return 0.5 * (entropy(p, m + eps) + entropy(q, m + eps))
def compute_mmd_rbf(X, Y, gamma=1.0):
	"""计算基于RBF核的最大均值差异 (MMD^2)"""
	XX = rbf_kernel(X, X, gamma=gamma)
	YY = rbf_kernel(Y, Y, gamma=gamma)
	XY = rbf_kernel(X, Y, gamma=gamma)
	return XX.mean() + YY.mean() - 2 * XY.mean()


def compute_distribution_distance(dataset1, dataset2, method='euclidean', sample_size=500, flatten=True):
	"""
	分析两个 dataset（一个类别）之间的分布差异度
	Args:
		dataset1, dataset2: torch Dataset，每个是一个类别
		method: 'euclidean', 'cosine', 'js', 'wasserstein', 'mmd'
		sample_size: 每类抽样样本数量
		flatten: 是否将图像展平为1D向量（默认为True）
	Returns:
		float: 差异度值
	"""
	loader1 = DataLoader(dataset1, batch_size=sample_size, shuffle=True)
	loader2 = DataLoader(dataset2, batch_size=sample_size, shuffle=True)

	imgs1, _ = next(iter(loader1))
	imgs2, _ = next(iter(loader2))

	if flatten:
		imgs1 = imgs1.view(imgs1.size(0), -1)
		imgs2 = imgs2.view(imgs2.size(0), -1)

	imgs1 = imgs1.numpy()
	imgs2 = imgs2.numpy()

	hist1, _ = np.histogram(imgs1, bins=100, range=(0, 1), density=True)
	hist2, _ = np.histogram(imgs2, bins=100, range=(0, 1), density=True)
	result = {'euclidean': euclidean(imgs1.mean(axis=0), imgs2.mean(axis=0)),
			  'cosine': cosine(imgs1.mean(axis=0), imgs2.mean(axis=0)), 'js': compute_js_divergence(hist1, hist2),
			  'wasserstein': ot.wasserstein_1d(imgs1.flatten(), imgs2.flatten()), 'mmd': compute_mmd_rbf(imgs1, imgs2, gamma=1.0)}
	return result

def evaluate_threshold(A, B, percentile=95):
	"""
	在A中找一个percentile阈值，在B中统计超过这个阈值的比例

	参数:
		A: list 或 np.array，ID样本得分（如Mahalanobis距离）
		B: list 或 np.array，OOD样本得分
		percentile: 阈值百分位数，默认95%

	返回:
		threshold: 阈值
		B_over_threshold_ratio: B中超过该阈值的比例
	"""
	A = np.array(A)
	B = np.array(B)

	threshold = np.percentile(A, percentile)
	B_count = np.sum(B > threshold)
	B_ratio = B_count / len(B)

	return threshold, B_ratio

from datasets import *
def test_pic_diff():
	train_data, test_data = get_dataset(2)

	train_class_datasets = split_by_class(train_data, 10)

	train_class_loaders = create_class_loaders(train_class_datasets, shuffle=False)

	test_class_datasets = split_by_class(test_data, 10)

	test_class_loaders = create_class_loaders(test_class_datasets, shuffle=False)

	ood = get_ood_cifar100(2)

	ood_class_datasets = split_by_class(ood, 100)

	# 创建测试集各类别DataLoader
	ood_class_loaders = create_class_loaders(ood_class_datasets, shuffle=False)

	print(compute_distribution_distance(train_class_loaders[0].dataset, test_class_loaders[0].dataset))
	print(compute_distribution_distance(train_class_loaders[0].dataset, ood_class_loaders[0].dataset))

from models import *
from sklearn import manifold


def get_ith_filepath(folder_path: str, i: int) -> str:
	if i == 1:
		i=0
	elif i == 10:
		i=1
	"""
    获取文件夹中第i个文件的完整路径 (按文件名排序)

    :param folder_path: 目标文件夹路径
    :param i: 文件索引 (从0开始)
    :return: 第i个文件的完整路径

    Raises:
        NotADirectoryError: 输入路径不是文件夹
        IndexError: 索引超出文件数量范围
        FileNotFoundError: 文件夹不存在
    """
	# 检查文件夹是否存在
	if not os.path.exists(folder_path):
		raise FileNotFoundError(f"文件夹不存在: {folder_path}")

	# 检查输入是否为目录
	if not os.path.isdir(folder_path):
		raise NotADirectoryError(f"输入路径不是文件夹: {folder_path}")
	# 获取文件夹下所有条目，并过滤出文件
	all_files = [
		entry for entry in os.listdir(folder_path)
		if os.path.isfile(os.path.join(folder_path, entry))
	]

	# 检查是否有文件
	if not all_files:
		raise FileNotFoundError(f"文件夹中没有文件: {folder_path}")
	# 按文件名排序 (升序)
	sorted_files = sorted(all_files)

	# 检查索引是否越界
	if i < 0 or i >= len(sorted_files):
		raise IndexError(f"索引 {i} 超出范围 (0-{len(sorted_files) - 1})")
	# 拼接完整路径
	target_file = sorted_files[i]
	full_path = os.path.join(folder_path, target_file)
	print(full_path)

	return full_path

def T_SNE():
	train_data, test_data = get_dataset(1)

	test_class_datasets = split_by_class(test_data, 10)

	test_class_loaders = create_class_loaders(test_class_datasets, shuffle=False)

	ood = get_ood_cifar100(1)

	ood_class_datasets = split_by_class(ood, 100)

	# 创建测试集各类别DataLoader
	ood_class_loaders = create_class_loaders(ood_class_datasets, shuffle=False)

	model = ResNet_18(BasicBlock, num_classes=10).to(device)
	model.load_state_dict(torch.load(get_ith_filepath(f'./ood/15', 7)))

	id_features, id_result = detect_ood_with_mahalanobis(model, test_class_loaders, test_class_loaders)
	ood_features, ood_result = detect_ood_with_mahalanobis(model, test_class_loaders, ood_class_loaders)
	threshold, B_ratio = evaluate_threshold(id_result, ood_result, 95)
	print(threshold)
	for i in id_result:
		print(i)
	print('------')
	for i in ood_result:
		print(i)
	exit(0)

	label = np.array([0] * len(id_features) + [1] * len(ood_features), dtype='uint8')

	normal_idxs = (label == 0)
	abnorm_idxs = (label == 1)

	data = torch.cat([id_features, ood_features]).numpy()

	tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(data)

	# tsne 归一化， 这一步可做可不做
	x_min, x_max = tsne.min(0), tsne.max(0)
	tsne_norm = (tsne - x_min) / (x_max - x_min)

	tsne_normal = tsne_norm[normal_idxs]
	tsne_abnormal = tsne_norm[abnorm_idxs]

	plt.figure(figsize=(8, 8))
	plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 1, color='red', label='ID数据集')
	# tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
	plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='green', label='OOD数据集')
	plt.legend(loc='upper left')
	plt.show()


def min_max_scale(data):
	if len(data) == 0:
		raise ValueError("输入列表不能为空")

	min_val = min(data)
	max_val = max(data)

	if max_val == min_val:
		# 处理所有元素相同的情况（避免除以0）
		return [0.0 for _ in data]

	scaled = [(x - min_val) / (max_val - min_val) for x in data]
	return scaled

def roc():
	"""
	y_true：真实标签
	y_score：模型预测分数
	pos_label：正样本标签，如“1”
	"""

	train_data, test_data = get_dataset(1)

	test_class_datasets = split_by_class(test_data, 10)

	test_class_loaders = create_class_loaders(test_class_datasets, shuffle=False)

	ood = get_ood_cifar100(1)

	ood_class_datasets = split_by_class(ood, 100)

	# 创建测试集各类别DataLoader
	ood_class_loaders = create_class_loaders(ood_class_datasets, shuffle=False)

	model = ResNet_18(BasicBlock, num_classes=10).to(device)
	model.load_state_dict(torch.load(f"./95%正确率模型.pth"))

	id_features, id_result = detect_ood_with_mahalanobis(model, test_class_loaders, test_class_loaders)
	ood_features, ood_result = detect_ood_with_mahalanobis(model, test_class_loaders, ood_class_loaders)

	pos_label = 0
	y_true = [0]*len(ood_result)+[1]*len(id_result)
	y_score = id_result + ood_result

	y_true = np.array(y_true)

	# 统计正样本和负样本的个数
	num_positive_examples = (y_true == pos_label).sum()
	num_negtive_examples = len(y_true) - num_positive_examples

	tp, fp = 0, 0
	tpr, fpr, thresholds = [], [], []
	score = max(y_score) + 1

	# 根据排序后的预测分数分别计算fpr和tpr
	for i in np.flip(np.argsort(y_score)):
		# 处理样本预测分数相同的情况
		if y_score[i] != score:
			fpr.append(fp / num_negtive_examples)
			tpr.append(tp / num_positive_examples)
			thresholds.append(score)
			score = y_score[i]

		if y_true[i] == pos_label:
			tp += 1
		else:
			fp += 1

	fpr.append(fp / num_negtive_examples)
	tpr.append(tp / num_positive_examples)
	thresholds.append(score)

	plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
	plt.rcParams['axes.unicode_minus'] = False  # 显示负号

	plt.plot(fpr, tpr)
	plt.axis("square")
	plt.xlabel("假正例率")
	plt.ylabel("真正例率")
	plt.title("25%正确率模型的ROC测试曲线")
	plt.show()

