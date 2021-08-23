import json, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import socket

import drn as models


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
    # insta_pw = string_splitted[1]
    d_insta_id = insta_id.replace("'", "")
    # d_insta_pw = insta_pw.replace("'", "")
    #print("소켓 ID : {}".format(d_insta_id))

    # 소켓을 닫습니다.
    client_socket.close()

    return d_insta_id


def create_dataset(test_path):
    image = []
    for test_file in os.listdir(test_path):
        if test_file[len(test_file)-3:] == 'jpg':
            image.append(test_path + '/' + test_file)
    return image


class Instagram_Dataset(Dataset):
    def __init__(self, img_data, img_path, transform=None):
        self.img_data = img_data
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        image = Image.open(self.img_data[index])
        # image = image.convert('RGB')
        image = image.resize((300, 300))
        if self.transform is not None:
            image = self.transform(image)
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
        # print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint)
        # print("=> loaded checkpoint '{}'".format(filepath))
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


if __name__ == '__main__':
    d_insta_id = "iseongmin_97"
    #d_insta_id = client_socket(d_insta_id)
    # print("소켓 실행")

    # 이미지 분류에 사용될 파일의 경로
    test_folder_path = 'C:/Users/pc/Desktop/캡스톤/pose/pose/src/main/resources/static/images/downloadImg/'
    # 모델 경로
    model_path = 'C:/Users/pc/Desktop/캡스톤/python/model/image_classification_drn_model-epoch30.pt'
    # 데이터셋 클래스 갯수
    top_k_classes = 20
    # 데이터셋 JSON파일 경로
    json_file = "C:/Users/pc/Desktop/캡스톤/python/dataset.json"
    device = "cuda"
    drn_model_name = "drn_c_26"
    # 이미지 분류에 사용될 이미지파일의 갯수
    batch_size = 1
    num_workers = 1

    # 사용되는 모델 DRN C 26 모델
    model = models.__dict__[drn_model_name]()
    # 신경망 데이터 병렬 처리
    model = torch.nn.DataParallel(model).cuda()

    # 테스트 폴더에 있는 이미지들을 데이터셋으로 구성
    image = create_dataset(test_folder_path)
    data = {}

    # 데이터셋에 있는 이미지 전처리
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                              std=(0.5, 0.5, 0.5))])

    # 테스트 데이터셋 구성
    dataset = Instagram_Dataset(image, test_folder_path, transform)

    # 테스트 데이터셋을 모델에 적용하기 위해 데이터로더형식으로 변환
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    # 미리 훈련된 모델을 불러옵니다.
    model = load_checkpoint(model_path, model)
    # 결과값을 보여줍니다.
    visualization(test_loader)
    #csv_file_path = 'C:/Users/pc/Desktop/캡스톤/python/crawling/Top50/#' + predict_class + '_most_used_tag.csv'
    predict_class = visualization(test_loader)
    csv_file_path = 'C:/Users/pc/Desktop/캡스톤/python/crawling/recommended_hashtag/#' + predict_class + '_Top_10.csv'
    csv = pd.read_csv(csv_file_path,
                      header=0,
                      names=['index', 'tag'],
                      usecols=['tag'])
    for row in range(10):
        print(csv.loc[row][0])

    print("end")
