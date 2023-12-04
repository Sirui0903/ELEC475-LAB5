import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader


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
                target = (position_x,position_y)
                self.data.append((filename, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, target = self.data[index]
        image_path = os.path.join(self.data_dir, filename)
        # image = cv2.imread(i  ``mage_path, cv2.IMREAD_COLOR)
        image = Image.open(image_path).convert('RGB')
        target = (target[0]*224/image.size[0],target[1]*224/image.size[1])
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.float32)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-d', metavar='display', type=str, help='[y/N]')
    Parser.add_argument('--img_size', type=int, default=224)
    args = Parser.parse_args()

    show_images = False
    if args.d != None:
        if args.d == 'y' or args.d == 'Y':
            show_images = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = './images'

    densenetModel = models.densenet121()
    num_features = densenetModel.classifier.in_features
    densenetModel.classifier = nn.Linear(num_features, out_features=2)
    densenetModel.load_state_dict(torch.load("./epoch40_batchsize8.pth"))
    densenetModel.to(device)
    densenetModel.eval()

    test_dataset = catDataset(data_dir=data_dir, mode='test', transform=transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    distances = []

    for image_tensor, true_coords in test_loader:
        image_tensor = image_tensor.to(device)
        true_coords_np = true_coords.squeeze().numpy()

        with torch.no_grad():
            predicted_coords = densenetModel(image_tensor).cpu().numpy()[0]

        distance = np.linalg.norm(true_coords_np - predicted_coords)
        distances.append(distance)

        if show_images:
            print("Predicted coords:", predicted_coords)
            print("True coords before squeeze:", true_coords)
            image_np = transforms.ToPILImage()(image_tensor.squeeze(0)).convert('RGB')
            image_np = np.array(image_np)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            true_coords = true_coords.squeeze().numpy()
            print("True coords after squeeze:", true_coords)

            true_x, true_y = true_coords.astype(int)
            pred_x, pred_y = predicted_coords.astype(int)

            image_np = cv2.circle(image_np, (true_x, true_y), 5, (0, 255, 0), -1)
            image_np = cv2.circle(image_np, (pred_x, pred_y), 5, (0, 0, 255), -1)
            print(f"Distance: {distance:.4f}")

            cv2.imshow('Predicted Image', image_np)

            key = cv2.waitKey(0)
            if key == ord('x'):
                break

    distances = np.array(distances)
    min_distance = distances.min()
    mean_distance = distances.mean()
    max_distance = distances.max()
    std_deviation = distances.std()

    # Print out the statistics
    print(f"Minimum Euclidean Distance: {min_distance:.4f}")
    print(f"Mean Euclidean Distance: {mean_distance:.4f}")
    print(f"Maximum Euclidean Distance: {max_distance:.4f}")
    print(f"Standard Deviation of Euclidean Distance: {std_deviation:.4f}")

main()