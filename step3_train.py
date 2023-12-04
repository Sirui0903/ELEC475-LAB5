
import timm

timm.list_models('swin*')

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import random

color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)


class RandomHorizontalFlipTransform:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if random.random() < self.probability:
            image = transforms.functional.hflip(image)
            target = (image.width - target[0], target[1])
        return image, target


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def mean_absolute_error(preds, labels):
    return torch.mean(torch.abs(preds - labels))


class catDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.mode = mode

        file_name = 'train_noses.txt' if mode == 'train' else 'test_noses.txt'
        with open(os.path.join(data_dir, file_name), 'r') as file:
            lines = file.readlines()
            for line in lines:
                # Skip empty lines or lines that do not contain a comma
                if not line.strip() or ',' not in line:
                    continue

                filename, nose_coords = line.strip().split(',', 1)
                nose_coords = nose_coords.strip('() "')
                position_x, position_y = (int(coord) for coord in nose_coords.split(','))
                target = (position_x, position_y)
                self.data.append((filename, target))

        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, target = self.data[index]
        image_path = os.path.join(self.data_dir, filename)
        image = Image.open(image_path).convert('RGB')

        # RandomHorizontalFlipTransform
        flip_transform = RandomHorizontalFlipTransform()
        image, target = flip_transform(image, target)

        # normalize
        target = (target[0] * 224 / image.width, target[1] * 224 / image.height)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_dir = './images'
# parser = argparse.ArgumentParser(description="")
# parser.add_argument('-epoch', type=int, default=40)
# parser.add_argument('-b', '--batch_size', type=int, default=128)
# parser.add_argument('-lr', type=float, default=0.0001)
# parser.add_argument('--lr_decay', type=float, default=1e-4)
# parser.add_argument('--img_size', type=int, default=224)
# args = parser.parse_args()

args = {
    'epoch': 40,
    'batch_size': 8,
    'lr': 0.001,
    'lr_decay': 1e-4,
    'img_size': 224
}

args = argparse.Namespace(**args)

# Create dataset
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.Lambda(lambda x: color_jitter(x)),
    transforms.ToTensor()
])

train_dataset = catDataset(data_dir=data_dir, mode='train', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_dataset = catDataset(data_dir=data_dir, mode='test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

densenetModel = models.densenet121(pretrained=True)
num_features = densenetModel.classifier.in_features  # get in features
densenetModel.classifier = nn.Linear(num_features, out_features=2)  # x and y

# resnetModel = models.resnet18(pretrained=True)
# num_features = resnetModel.fc.in_features

# resnetModel.fc = nn.Sequential(
#     nn.Linear(num_features, 2)
# )

# transformModel = timm.create_model('swin_s3_tiny_224',pretrained=True)
# num_features = transformModel.head.fc.in_features
# transformModel.head.fc = nn.Linear(num_features, out_features=2, bias=True)

densenetModel.to(device)

criterion = nn.L1Loss()
optimizer = optim.SGD(densenetModel.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROenPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

train_losses = []
test_losses = []

for epoch in range(args.epoch):
    densenetModel.train()
    # adjust_learning_rate(optimizer, iteration_count=args.batch_size)
    running_loss = 0.0
    tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epoch}, Training")
    for inputs, labels in tqdm_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = densenetModel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        tqdm_train_loader.set_postfix(loss=running_loss / len(tqdm_train_loader), refresh=True)

    train_losses.append(running_loss / len(train_loader))

    '''Validation'''

    densenetModel.eval()
    test_loss = 0.0

    # Use tqdm for progress bar
    tqdm_test_loader = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{args.epoch}, Testing")
    with torch.no_grad():
        for inputs, labels in tqdm_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = densenetModel(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            tqdm_test_loader.set_postfix(loss=test_loss / len(tqdm_test_loader), refresh=True)

    scheduler.step(test_loss / len(test_loader))
    test_losses.append(test_loss / len(test_loader))

    tqdm_train_loader.close()
    tqdm_test_loader.close()

torch.save(densenetModel.state_dict(), f'epoch{args.epoch}_batchsize{args.batch_size}.pth')

# Plot the loss over epochs

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./epoch{args.epoch}_batchsize{args.batch_size}.png')
plt.show()