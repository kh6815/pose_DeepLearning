import json, os, math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import socket

import drn as models


class DRN_MODEL(nn.Module):
    def __init__(self, model_name, classes, use_torch_up=False):
        super(DRN_MODEL, self).__init__()
        model = models.__dict__.get(model_name)(num_classes=20)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Instagram_Dataset(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_data = img_data
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        img_name = self.img_path + self.img_data[index]
        image = Image.open(img_name)
        # image = image.convert('RGB')
        image = image.resize((300, 300))
        if self.transform is not None:
            image = self.transform(image)
        return image


def create_dataset(test_path):
    image = []
    for test_file in os.listdir(test_path):
        if test_file[:len(test_file)-4] == d_insta_id:
            image.append(test_file)
    return image


# Load class_to_name json file
def load_json(json_file):
    with open(json_file, 'r') as f:
        hashtags_to_name = json.load(f)
        return hashtags_to_name


# Function for loading the model checkpoint
def load_checkpoint(filepath, model):
    # optionally resume from a checkpoint
    if os.path.isfile(filepath):
        #print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint)
        #print("=> loaded checkpoint '{}'".format(filepath))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))
    return model


def visualization(data_loader):
    # get some random training images
    data_iter = iter(data_loader)
    images = data_iter.next()
    instagram_types = {0: "birthday", 1: "cafe", 2: "camping", 3: "cat", 4: "chicken",
                       5: "couple", 6: "dog", 7: "exercise", 8: "fashion", 9: "flower",
                       10: "hiking", 11: "nail_art", 12: "parenting", 13: "reading", 14: "selfie",
                       15: "spring", 16: "summer", 17: "travel", 18: "tteokbokki", 19: "wedding"}
    # Viewing data examples used for training
    fig, axis = plt.subplots(2, 5, figsize=(15, 10))
    with torch.no_grad():
        model.eval()
        idx = 1
        for ax, image in zip(axis.flat, images):
            # ax.imshow(img_display(image))  # add image
            image_tensor = image.unsqueeze_(0)
            output_ = model(image_tensor)
            output_ = output_.argmax()
            k = output_.item()
            #print("{}번째 사진은 {}입니다.".format(idx, str(instagram_types[output_.item()])))
            print("title : {}".format(str(instagram_types[output_.item()])));
            idx = idx + 1
            # ax.set_title(str(instagram_types[output_.item()]) + ":" + str(k))  # add label
    #plt.savefig('./experiment/test_dataset.png')
    #plt.show()
    return str(instagram_types[output_.item()])


def img_display(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def client_socket(d_insta_id) :
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST = '127.0.0.1'
    # 서버에서 지정해 놓은 포트 번호입니다.
    PORT = 8000

    # 소켓 객체를 생성합니다.
    # 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용합니다.
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 지정한 HOST와 PORT를 사용하여 서버에 접속합니다.
    client_socket.connect((HOST, PORT))

    # 메시지를 수신합니다.
    data = client_socket.recv(1024)
    string_splitted = repr(data.decode()).split()
    insta_id = string_splitted[0]
    insta_pw = string_splitted[1]
    d_insta_id = insta_id.replace("'", "")
    d_insta_pw = insta_pw.replace("'", "")
    #print("소켓 ID : {}, PW : {}".format(d_insta_id, d_insta_pw))

    #print('Received from', "d_insta_id-----------------------------")
    #print('Received from', d_insta_id)
    #print('Received from', d_insta_pw)
    #print('Received', repr(data.decode()))

    # 소켓을 닫습니다.
    client_socket.close()

    return d_insta_id

if __name__ == '__main__':
    d_insta_id =""
    d_insta_id = client_socket(d_insta_id)
   #print("소켓 나옴")

    test_folder_path = 'C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/images/downloadImg/'
    model_path = 'C:/Users/pc/Desktop/캡스톤/python/model/image_classification_drn_model-epoch10.pt'
    top_k_classes = 20
    json_file = "C:/Users/pc/Desktop/캡스톤/python/dataset.json"
    device = "cuda"
    drn_model_name = "drn_c_26"
    batch_size = 10
    num_workers = 1

    #print(d_insta_id)

    model = models.__dict__[drn_model_name]()

    model = torch.nn.DataParallel(model).cuda()

    image = create_dataset(test_folder_path)

    data = {}

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                              std=(0.5, 0.5, 0.5))])

    dataset = Instagram_Dataset(image, test_folder_path, transform)
    # print(dataset.img_data)
    # print(dataset.img_path)
    # print(dataset.transform)

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )
    # print(test_loader.dataset.__len__())
    # print(test_loader.dataset.__getitem__(0))

    # Load pretrained network
    model = load_checkpoint(model_path, model)
    # print(model)

    predict_class = visualization(test_loader)
    csv_file_path = 'C:/Users/pc/Desktop/캡스톤/python/crawling/recommended_hashtag/#' + predict_class + '_Top_10.csv'
    csv = pd.read_csv(csv_file_path,
                      header=0,
                      names=['index', 'tag'],
                      usecols=['tag'])
    #hashtag_list = ""
    for row in range(10):
        #hashtag_list = hashtag_list + csv.loc[row][0]
        print(csv.loc[row][0])

    #print(hashtag_list)
    print("end")