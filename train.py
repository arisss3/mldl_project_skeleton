import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import wandb

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# root/{class}/x001.jpg

tiny_imagenet_dataset_train = ImageFolder(root='dataset/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='dataset/tiny-imagenet-200/val', transform=transform)

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

val_images_path = 'dataset/tiny-imagenet-200/val/images'
if os.path.exists(val_images_path):  # Check if the directory exists
    with open('dataset/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'dataset/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'{val_images_path}/{fn}', f'dataset/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree(val_images_path)  # Remove the directory after processing




def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()  # Move to GPU

        optimizer.zero_grad()
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

    return train_loss, train_accuracy

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()  # Move to GPU

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

best_acc = 0
num_epochs = 7

for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model saved with accuracy: {best_acc:.2f}%")



wandb.init(project='mldl_project_skeleton')

config = wandb.config
config.learning_rate = 0.01

for i in range(10):
  wandb.log({"loss": loss})