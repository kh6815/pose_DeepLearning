# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os.path import join
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
# Neural networks can be constructed using the torch.nn package.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms


def create_dataset(dataset_path, image, label):
    for category in os.listdir(dataset_path):
        for folder in os.listdir(join(dataset_path, category)):
            for file in os.listdir(join(dataset_path, category, folder)):
                if file.split('.')[1] == 'jpg':
                    image.append(file)
                    label.append(category + '/' + folder)
    return image, label


def split_dataset(data):
    validation_split = .3
    shuffle_dataset = True
    random_seed = 42
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


class Instagram_Dataset(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_data = img_data
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img_name = join(self.img_path, self.img_data.loc[index, 'labels'],
                        self.img_data.loc[index, 'Images'])
        image = Image.open(img_name)
        # image = image.convert('RGB')
        image = image.resize((300, 300))
        label = torch.tensor(self.img_data.loc[index, 'encoded_labels'])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def img_display(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.4)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.dropout(self.batchnorm2(self.pool(x)))
        x = self.batchnorm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.conv4(x))
        x = x.view(-1, 64 * 5 * 5)  # Flatten layer
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def train_model(train_loss, val_loss, train_acc, val_acc):
    n_epochs = 12
    valid_loss_min = np.Inf
    total_step = len(train_loader)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_loader):
            data_, target_ = data_.to(device), target_.to(device)  # on GPU
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            print(data_.shape)
            print(data_)
            print(outputs.shape)
            print(outputs)
            print(target_.shape)
            print(target_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')
        valid_loss_min = evaluate_model(valid_loss_min, val_loss, val_acc)


def evaluate_model(valid_loss_min, val_loss, val_acc):
    batch_loss = 0
    total_t = 0
    correct_t = 0
    with torch.no_grad():
        model.eval()
        for data_t, target_t in (validation_loader):
            # data_t, target_t = data_t.to(device), target_t.to(device)# on GPU
            outputs_t = model(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(validation_loader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
        # Saving the best weight
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'model_classification_tutorial.pt')
            print('Detected network improvement, saving current model')
    model.train()
    return valid_loss_min


def show_graph(graph_type, train, val):
    plt.figure(figsize=(20, 10))
    plt.title("Train - Validation " + graph_type)
    plt.plot(train, label='train')
    plt.plot(val, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel(graph_type, fontsize=12)
    plt.legend(loc='best')


def visualization(graph_type, data_loader):
    # get some random training images
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    instagram_types = {0: 'animal/cat',
                       1: 'animal/dog',
                       2: 'daily/cafe',
                       3: 'hobby/travel',
                       4: 'nature/spring'}
    # Viewing data examples used for training
    fig, axis = plt.subplots(3, 5, figsize=(15, 10))
    if graph_type == "train":
        for i, ax in enumerate(axis.flat):
            with torch.no_grad():
                image, label = images[i], labels[i]
                ax.imshow(img_display(image))  # add image
                ax.set(title=f"{instagram_types[label.item()]}")  # add label
        plt.savefig('./dataset_elements.png')
        plt.show()
    elif graph_type == "val":
        with torch.no_grad():
            model.eval()
            for ax, image, label in zip(axis.flat, images, labels):
                ax.imshow(img_display(image))  # add image
                image_tensor = image.unsqueeze_(0)
                output_ = model(image_tensor)
                output_ = output_.argmax()
                k = output_.item() == label.item()
                ax.set_title(str(instagram_types[label.item()]) + ":" + str(k))  # add label
        plt.savefig('./evaluate_model.png')
        plt.show()


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    # print(device)

    dataset_path = './dataset'

    image = []
    label = []
    data = {}
    data["Images"], data["labels"] = create_dataset(dataset_path, image, label)

    label_encoder = LabelEncoder()
    data['encoded_labels'] = label_encoder.fit_transform(data['labels'])
    data = pd.DataFrame(data)
    # data.to_csv("./dataset.csv")
    # print(data.head())

    batch_size = 128
    num_workers = 8

    # Creating data indices for training and validation splits:
    # tr, val = train_test_split(data.label, stratify=data.label, test_size=0.1)

    train_indices, val_indices = split_dataset(data)
    # print(train_indices)
    # print(val_indices)

    # train_indices is equivalent to list(tr.index)
    # val_indices is equivalent to list(val.index)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                              std=(0.5, 0.5, 0.5))])

    dataset = Instagram_Dataset(data, dataset_path, transform)
    # print(dataset.img_data)
    # print(dataset.img_path)
    # print(dataset.transform)
    # print(dataset.__getitem__(train_indices[0]))


    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               sampler=train_sampler)

    # print(train_loader.dataset.__getitem__(train_indices[0]))

    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    sampler=train_sampler)

    # visualization("train", train_loader)

    # model = Net()  # On CPU
    model = torch.nn.DataParallel(Net()).cuda()  # On GPU
    # print(model)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4)

    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    train_model(train_loss, val_loss, train_acc, val_acc)

    show_graph("Loss", train_loss, val_loss)
    show_graph("Accuracy", train_acc, val_acc)

    # Importing trained Network with better loss of validation
    model.load_state_dict(torch.load('model_classification_tutorial.pt'))

    visualization("val", validation_loader)