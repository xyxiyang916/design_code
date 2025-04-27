from datasets import *
from models import *
from train import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, test = get_dataset()

test_class_datasets = split_by_class(test, 10)

test_class_loaders = create_class_loaders(test_class_datasets, shuffle=False)

ood = get_ood_cifar100()

ood_class_datasets = split_by_class(ood, 100)

# 创建测试集各类别DataLoader
ood_class_loaders = create_class_loaders(ood_class_datasets, shuffle=False)

model = ResNet_18(BasicBlock, num_classes=10).to(device)
model.load_state_dict(torch.load(f"models/end/end_model_1.pth"))

#extract_features(model, test_class_loaders[0])

result = detect_ood_with_mahalanobis(model, test_class_loaders, ood_class_loaders)

print(result)

#save_cifar10_images_labels(ood_class_loaders[0].dataset)