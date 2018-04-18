#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import numpy as np
import math
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from crnn import CRNN
from utils import *


def train(root, start_epoch, epoch_num, letters, batch_size,
          model=None, lr=0.1, data_size=None):
    """
    Train CRNN model

    Args:
        root (str): Root directory of dataset
        start_epoch (int): Epoch number to start
        epoch_num (int): Epoch number to train
        letters (str): Letters contained in the data
        batch_size (int): Size of each batch
        model (CRNN, optional): CRNN model (default: None)
        lr (float, optional): Coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        data_size (int, optional): Size of data to use (default: All data)

    Returns:
        CRNN: Trained CRNN model
    """

    # load data
    trainloader = Loader(root, batch_size=batch_size,
                         training=True, data_size=data_size)
    # use gpu or not
    use_cuda = torch.cuda.is_available()
    if not model:
        # create a new model if model is None
        model = CRNN(1, len(letters) + 1)
    # loss function
    criterion = CTCLoss()
    # Adadelta
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====   Training..   ====')
    # .train() has any effect on Dropout and BatchNorm.
    model.train()
    for epoch in range(start_epoch, start_epoch + epoch_num):
        print('----    epoch: %d    ----' % (epoch, ))
        loss_sum = 0
        for i, (img, label) in enumerate(trainloader):
            label, label_length = labeltransformer.encode(label)
            if use_cuda:
                img = img.cuda()
            img, label = Variable(img), Variable(label)
            label_length = Variable(label_length)
            optimizer.zero_grad()
            # put images in
            outputs = model(img)
            output_length = Variable(torch.IntTensor(
                [outputs.size(0)]*outputs.size(1)))
            # calc loss
            loss = criterion(outputs, label, output_length, label_length)
            if (np.isnan(loss.data[0])):
                # print(loss)
                # print(sum(np.isnan(label.data.view(-1))))
                # print(sum(np.isnan(img.data.view(-1))))
                # print(sum(np.isnan(outputs.data.view(-1))))
                # print(sum(np.isnan(output_length.data.view(-1))))
                # print(sum(np.isnan(label_length.data.view(-1))))
                # print(sum(np.isnan(outputs.data.view(-1))))
                # allsum = 0
                # for k in model.parameters():
                #     allsum += sum(np.isnan(k.data.view(-1)))
                # print(allsum)
                continue
            # update
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))
    print('Finished Training')
    return model


def test(root, model, letters, batch_size, data_size=None):
    """
    Test CRNN model

    Args:
        root (str): Root directory of dataset
        model (CRNN, optional): trained CRNN model
        letters (str): Letters contained in the data
        batch_size (int): Size of each batch
        data_size (int, optional): Size of data to use (default: All data)
    """

    # load data
    testloader = Loader(root, batch_size=batch_size,
                         training=False, data_size=data_size)
    # use gpu or not
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)

    print('====    Testing..   ====')
    # .eval() has any effect on Dropout and BatchNorm.
    model.eval()
    correct = 0
    for i, (img, origin_label) in enumerate(testloader):
        if use_cuda:
            img = img.cuda()
        img = Variable(img)

        outputs = model(img)  # length × batch × num_letters
        outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
        outputs = labeltransformer.decode(outputs.data)
        correct += sum([out == real for out,
                        real in zip(outputs, origin_label)])
    # calc accuracy
    print('test accuracy: ', correct / 30, '%')


def main(training=True):
    """
    Main

    Args:
        training (bool, optional): If True, train the model, otherwise test it (default: True)
    """

    model_path = 'crnn.pth'
    with open('letters.txt', 'r') as fp:
        letters = fp.readline()
    root = 'data/'
    if training:
        model = CRNN(1, len(letters) + 2)
        start_epoch = 0
        epoch_num = 50
        lr = 0.00022
        # if there is pre-trained model, load it
        if os.path.exists(model_path):
            print('Pre-trained model detected.\nLoading model...')
            model.load_state_dict(torch.load(model_path))
        model = train(root, start_epoch, epoch_num, letters, 32,
                    model=model, lr=lr, data_size=1000)
        test(root, model, letters, 32, 500)
        # save the trained model for training again
        torch.save(model.state_dict(), model_path)
    else:
        model = CRNN(1, len(letters) + 1)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        test(root, model, letters, 32, 500)


if __name__ == '__main__':
    main(training=True)