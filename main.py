#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import Levenshtein
import pickle
import argparse
import numpy as np
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from crnn import CRNN
from utils import *


def train(root, model_path, letters, batch_size, epoch_num, lr=0.1,
          decay=5e-4, data_size=None, optim='adadelta', workers=2):
    """
    Train CRNN model
    Args:
        root (str): Root directory of dataset
        model_path (str): Path to save/load model
        letters (str): Letters contained in the data
        batch_size (int): Size of each batch
        epoch_num (int): Epoch number to train
        lr (float, optional): Coefficient that scale delta before it is applied
            to the parameters (default: 0.1)
        decay (float, optional): Weight decay in the model (default: 5e-4)
        data_size (int, optional): Size of data to use (default: All data)
        optim (str): Name of optim method (default: 'rmsprop')
        workers (int): Workers number to load data of each ratio (default: 2)
    Returns:
        CRNN: Trained CRNN model
    """

    model = CRNN(1, len(letters) + 2)
    if optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), weight_decay=decay)
    else:
        optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=decay)
    start_epoch = 0
    if os.path.exists(model_path):
        print('Pre-trained model detected.\nLoading model...')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        # if optimizer != None:
        optimizer.load_state_dict(checkpoint['optim'])
        if lr != None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        start_epoch = checkpoint['epoch']
    else:
        if lr == None:
            lr = 0.001 if optim == 'rmsprop' else (0.0001 if optim == 'adam' else 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    trainloader = Loader(root, batch_size=batch_size, training=True,
                         data_size=data_size, workers=workers)
    labeltransformer = LabelTransformer(letters)
    criterion = CTCLoss()

    # use gpu or not?
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    print('====   Training..   ====')
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

            outputs = model(img)
            output_length = Variable(torch.IntTensor(
                [outputs.size(0)]*outputs.size(1)))

            loss = criterion(outputs, label, output_length, label_length)
            if np.isnan(loss.data[0]) or abs(loss.data[0]) == float('inf'):
                continue
            loss.backward()
            optimizer.step()
            loss_sum += loss.data[0]
        print('loss = %f' % (loss_sum, ))

    checkpoint = {
        'model':model.state_dict(),
        'optim':optimizer.state_dict(),
        'epoch':start_epoch+epoch_num
    }
    torch.save(checkpoint, model_path)
    print('Model saved')


def test(root, model_path, letters, batch_size, data_size=None, workers=2):
    """
    Test CRNN model
    Args:
        root (str): Root directory of dataset
        model_path (str): Path to save/load model
        letters (str): Letters contained in the data
        batch_size (int): Size of each batch
        data_size (int, optional): Size of data to use (default: All data)
        workers (int): Workers number to load data of each ratio (default: 2)
    """

    if os.path.exists(model_path):
        model = CRNN(1, len(letters) + 2)
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        print('***** No model detected in %s! *****' % (model_path,))
        return
    testloader = Loader(root, batch_size=batch_size, training=False,
                        data_size=data_size, workers=workers)
    trainloader = Loader(root, batch_size=batch_size, training=True,
                        data_size=data_size, workers=workers)
    loaders = [testloader, trainloader]
    # use gpu or not
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")
    # get encoder and decoder
    labeltransformer = LabelTransformer(letters)
    fp = open('result.txt', 'w')

    print('====    Testing..   ====')
    # .eval() has any effect on Dropout and BatchNorm.
    model.eval()
    for j in range(len(loaders)):
        correct = 0
        total = 0
        ratio_sum = 0
        for i, (img, origin_label) in enumerate(loaders[j]):
            if use_cuda:
                img = img.cuda()
            img = Variable(img)

            outputs = model(img)  # length × batch × num_letters
            outputs = outputs.max(2)[1].transpose(0, 1)  # batch × length
            outputs = labeltransformer.decode(outputs.data)
            label_pairs = list(zip(outputs, origin_label))
            correct += sum([out == real for out, real in label_pairs])
            if j == 0:
                for out, real in label_pairs:
                    fp.write(''.join((real, '\t\t', out, '\n')))
            ratio_sum += sum([Levenshtein.ratio(out, real)
                              for out, real in zip(outputs, origin_label)])
            total += len(origin_label)
            # calc accuracy
        if j == 0:
            print('Test accuracy: ', correct / total * 100, '%')
            print('Levenshtein ratio: ', ratio_sum / total * 100, '%')
        else:
            print('Accuracy on train data: ', correct / total * 100, '%')
            print('Levenshtein ratio on train data: ',
                  ratio_sum / total * 100, '%')
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Whether to test directly (default is training)')
    parser.add_argument('--root', default='data/', help='path to dataset (default="data/")')
    parser.add_argument('--model_path', default='models/crnn.pth', help='path to model to save (default="models/crnn.pth")')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default=2)')
    parser.add_argument('--batchsize', type=int, default=64, help='input batch size (default=64)')
    parser.add_argument('--data_size', type=int, default=None, help='input data size (default all data)')
    parser.add_argument('--epoch_num', type=int, default=50, help='number of epochs to train for (default=50)')
    parser.add_argument('--check_epoch', type=int, default=10, help='epoch to save and test (default=10)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate for optim (default=no change or 0.1)')
    parser.add_argument('--decay', type=float, default=5e-4, help='weight decay for optim (default=no change or 5e-4)')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is adadelta)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is adadelta)')
    parser.add_argument('--rmsprop', action='store_true', help='Whether to use rmsprop (default is adadelta)')
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(os.path.dirname(opt.model_path)):
        os.makedirs(os.path.dirname(opt.model_path))
    with open('letters.txt', 'r') as fp:
        letters = fp.readline()

    if opt.adam:
        optim = 'adam'
    elif opt.rmsprop:
        optim = 'rmsprop'
    else:
        optim = 'adadelta'

    for i in range(opt.epoch_num // opt.check_epoch):
        if not opt.test:
            train(opt.root, opt.model_path, letters, opt.batchsize,
                  opt.check_epoch, lr=opt.lr, decay=opt.decay,
                  data_size=opt.data_size, optim=optim, workers=opt.workers)
            opt.lr = None
        test(opt.root, opt.model_path, letters, opt.batchsize,
             data_size=opt.data_size, workers=opt.workers)
        torch.cuda.empty_cache()
        if opt.test:
            break
