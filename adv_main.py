#
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from attackers import FGSM, BIM, DeepFool
import torchvision.transforms as transforms
from PIL import Image
from GenerateData import MyDataset
import sys, os, time
from Log import Logger
import torchvision.models as models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
from scipy import optimize
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from GenerateData import MyDataset
from Binary2img import createGreyScaleImage
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")


# model
pretrained_model = "./densenet121_model.mdl"

model = models.densenet121(pretrained=False)
model.features.conv0 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)
for param in model.parameters():
    param.requires_grad = True
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)


# cpu
model = model.to(device)
model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
model.eval()


def list2img(imglist):
    """
    
    """
    img = createGreyScaleImage(imglist)
    img.save("test.jpg")  #

    transform = transforms.Compose([transforms.ToTensor(),])  #

    img = transform(img)

    return img


def create_real_sample_from_adv(x_init, new_file_path, rate, flag) -> bytearray:
    """
   
    """
    if rate == 0.05 or rate == 0.15:
        x_real = x_init
        x_real_adv = b"".join([bytes([i]) for i in x_real])

        new_file_path = "./adv_data/adv_data_" + str(flag) + "/adv_" + new_file_path

        if new_file_path:
            with open(new_file_path, "wb") as f:
                f.write(x_real_adv)


def function_liner(x, a, b):  #
    return a * x + b


def least_squares(y):
    """
    """
    x = np.arange(0, len(y))
    popt, pcov = optimize.curve_fit(function_liner, x, y)
    a = popt[0]
    b = popt[1]

    return a


def save_list(least_squares_list, num, logits_distance, squares_length):
    """
    """
    if num < squares_length:
        least_squares_list[num] = logits_distance
        a = 0
    if num >= squares_length:
        least_squares_list.pop(0)
        least_squares_list.append(logits_distance)
        a = least_squares(least_squares_list)
        # print("==============ERROR======[",a,"[==============ERROR======")

    return least_squares_list, a


def main(num_i, rate, flag):

    batchsize = 64
    CLIP_MAX = 1
    CLIP_MIN = 0
    resize = 224
    squares_length = 20

    transformed_trainset = MyDataset(
        "./data", resize, mode="testdata", flag=flag, rate=rate
    )

    trainloader = DataLoader(
        dataset=transformed_trainset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=64,
    )

    attacker = FGSM(eps=0.1, clip_max=CLIP_MAX, clip_min=CLIP_MIN)

    # 获取初始
    x, y, indexeslist, filename = transformed_trainset.__getitem__(num_i)

    x_temp = x[::]
    num_flag = "fail"
    squares_flag = False

    least_squares_list = [0] * squares_length
    a_list = [0] * squares_length

    logits_distance = 0
    for ij in range(0, 500):
        if squares_flag or logits_distance < -4:
            continue

        x = list2img(x)
        c, h, w = x.shape
        x = transforms.Resize((resize, resize))(x)

        c_flag = 0
        x_c = x.clone()
        for j in range(0, 100):
            if c_flag == 1:
                continue
            adv_x = attacker.generate(model, x_c, y)
            x_c = adv_x.clone()
            adv_nx = torch.unsqueeze(adv_x, 0)
            with torch.no_grad():
                logits = model(adv_nx)
                y_flag = logits.argmax(dim=1)[0]
                if y_flag != y:
                    c_flag = 1

        adv_nx = torch.unsqueeze(adv_x, 0)
        nx = torch.unsqueeze(x, 0)

        with torch.no_grad():
            logits = model(nx)
            print("x:", logits)

            logits_distance = logits.numpy()[0][1] - logits.numpy()[0][0]
            least_squares_list, a = save_list(
                least_squares_list, ij, logits_distance, squares_length
            )
            if ij > squares_length:
                a_list, aa = save_list(a_list, ij - squares_length, a, squares_length)
                if ij > 2 * squares_length:
                    if abs(aa) < 0.001:
                        squares_flag = True

            yy = logits.argmax(dim=1)[0]
            if yy == 0 and logits_distance < -4:
                if ij == 0:
                    num_flag = "zero"
                else:
                    num_flag = "success"

                print("====", ij)
                create_real_sample_from_adv(x_temp, filename, rate, flag)

            logits = model(adv_nx)
            print("adv_x:", logits)

        adv_x = transforms.Resize((h, w))(adv_x)
        adv_x = torch.squeeze(adv_x, 0).numpy()
        adv_x = list((adv_x.reshape(h * w) * 255).astype(int))

        select1 = []
        for i in range(0, len(indexeslist)):
            select1.append(adv_x[indexeslist[i]])
        for i in range(0, len(indexeslist)):
            x_temp[indexeslist[i]] = select1[i]

        x = x_temp[::]

    if yy == 1:
        create_real_sample_from_adv(x_temp, filename, rate, flag)

    return num_flag


if __name__ == "__main__":

    result_list = {"fail": 0, "zero": 0, "success": 0}
    rate = 0.1
    flag = "section"
    print("===", "model:", pretrained_model, "rate:", rate, "funtion:", flag, "===")

    for j in range(0, 500):

        num_flag = main(j, rate, flag)
        result_list[num_flag] = result_list[num_flag] + 1
        print("============ERROR===============", j, "============ERROR===============")
        print(result_list)

