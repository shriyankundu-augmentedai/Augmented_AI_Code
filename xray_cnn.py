import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# =====================================
# DEVICE
# =====================================

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# =====================================
# TRANSFORMS
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================
# DATASETS
# =====================================

train_dataset = datasets.ImageFolder(
    "xray_cnn/chest_xray/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "xray_cnn/chest_xray/test",
    transform=transform
)

print("Classes:", train_dataset.classes)

# =====================================
# DATALOADERS
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# =====================================
# VISUALIZE DATA
# =====================================

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(
    2,
    5,
    figsize=(10, 5)
)

for i, ax in enumerate(axes.flat):

    img = images[i].permute(1, 2, 0)

    ax.imshow(img, cmap="gray")

    ax.set_title(
        train_dataset.classes[
            labels[i]
        ]
    )

    ax.axis("off")

plt.tight_layout()
plt.show()

# =====================================
# LOAD PRETRAINED RESNET18
# =====================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)

# =====================================
# REPLACE FINAL LAYER
# =====================================

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

print(model)

# =====================================
# PARAMETER COUNT
# =====================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Total Parameters: "
    f"{total_params:,}"
)

# =====================================
# LOSS + OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

# =====================================
# TRAIN
# =====================================

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" Loss: {avg_loss:.4f}"
    )

# =====================================
# TEST
# =====================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

accuracy = (
    100 * correct / total
)

print(
    f"\nTest Accuracy: "
    f"{accuracy:.2f}%"
)

# =====================================
# SAVE MODEL
# =====================================

torch.save(
    model.state_dict(),
    "pneumonia_resnet18.pth"
)

print(
    "Saved as pneumonia_resnet18.pth"
)

# =====================================
# DISPLAY PREDICTIONS
# =====================================

images, labels = next(
    iter(test_loader)
)

images_gpu = images.to(device)

with torch.no_grad():

    outputs = model(images_gpu)

    preds = outputs.argmax(
        1
    ).cpu()

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 6)
)

for i, ax in enumerate(
    axes.flat
):

    img = images[i].permute(
        1,
        2,
        0
    )

    predicted = train_dataset.classes[
        preds[i]
    ]

    actual = train_dataset.classes[
        labels[i]
    ]

    ax.imshow(
        img,
        cmap="gray"
    )

    ax.set_title(
        f"P:{predicted}\nA:{actual}",
        fontsize=8
    )

    ax.axis("off")

plt.tight_layout()
plt.show()