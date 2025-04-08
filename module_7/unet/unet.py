import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # Без GUI, рендеринг в файл
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 100
BATCH_SIZE = 4
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_paths = {
    "train": ("dataset/train/images", "dataset/train/masks"),
    "val": ("dataset/val/images", "dataset/val/masks"),
    "test": ("dataset/test/images", "dataset/test/masks")
}


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, img_name


# Трансформации
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def get_dataloader(split):
    img_dir, mask_dir = dataset_paths[split]
    dataset = SegmentationDataset(img_dir, mask_dir, transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


dataloaders = {split: get_dataloader(split) for split in ["train", "val", "test"]}


# Функция двойной свёртки
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = double_conv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = double_conv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        c1 = self.down1(x) 
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        bm = self.bottleneck(p4) 

        u4 = self.up4(bm) 
        u4 = torch.cat([u4, c4], dim=1)  
        c5 = self.conv4(u4)

        u3 = self.up3(c5)  
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.conv3(u3)

        u2 = self.up2(c6)  
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.conv2(u2) 

        u1 = self.up1(c7)  
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.conv1(u1)  

        out = self.out_conv(c8) 
        return torch.sigmoid(out)


# Обучение
model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_losses = [], []
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks, filename in dataloaders["train"]:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(dataloaders["train"]))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, filename in dataloaders["val"]:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    val_losses.append(val_loss / len(dataloaders["val"]))

    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

# Сохраняем
torch.save(model, "unet_model3.pth")

# График лоссов
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot3.png")

# предсказаний
prediction_dir = "prediction"
os.makedirs(prediction_dir, exist_ok=True)

# сохранение предсказаний из теста
model.eval()
iou_scores = []
with torch.no_grad():
    for images, masks, filenames in dataloaders["test"]:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        predicted_masks = (outputs > 0.5).float()

        # Вычисление IoU
        intersection = (predicted_masks * masks).sum()
        union = predicted_masks.sum() + masks.sum() - intersection
        iou = intersection / (union + 1e-6)
        iou_scores.append(iou.item())

        # Сохранение предсказанной маски
        for img, filename in zip(predicted_masks, filenames):
            pred_mask_np = img.squeeze().cpu().numpy() * 255
            pred_mask_pil = Image.fromarray(pred_mask_np.astype(np.uint8))
            pred_mask_pil.save(os.path.join(prediction_dir, filename))

iou_mean = np.mean(iou_scores)
print(f"IoU на тестовой выборке: {iou_mean:.4f}")

# График распределения IoU
plt.figure()
plt.hist(iou_scores, bins=20, edgecolor='black')
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.title("IoU Distribution on Test Set")
plt.savefig("iou_plot.png")
