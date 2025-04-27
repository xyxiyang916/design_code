import torch
from torch import nn, optim


def train(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 学习率衰减

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
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

    print(f"Train Loss: {total_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.2f}%")
    scheduler.step()  # 更新学习率

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
    for label, loader in ood_dataloader.items():
        if len(loader.dataset) == 0:
            continue
        # 获取一个类别的特征列表
        feats = extract_features(model, loader)
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
    return ood_scores

def test(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)