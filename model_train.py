# -*- coding:utf-8 -*-
"""
作者：于越
日期：2023年10月13日
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from models.mymodel import MSIF
from models.util import netDataset
from tqdm import trange
from scipy.io import savemat

os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'


# 用测试集评估模型的训练好坏
def eval(model, test_loaderv, test_laodera):
    eval_loss = 0.0
    total_acc = 0.0
    model.eval()
    for batch1, batch2 in zip(test_loaderv, test_laodera):
        #        batch = tuple(t.to(args.device) for t in batch)
        xv, y = batch1
        xa, _ = batch2
        xv = xv.to(device)
        xa = xa.to(device)
        y = y.to(device)
        with torch.no_grad():
            logitsb, logits, _, _, _ = model(xv, xa)  # model返回的是（bs,num_classes)
            batch_loss = loss_function(logits, y)
            # 记录误差
            eval_loss += batch_loss.item()
            # 记录准确率
            _, preds = logits.max(1)
            num_correct = (preds == y).sum().item()
            total_acc += num_correct
    loss = eval_loss / len(test_loaderv)
    acc = total_acc / (len(test_loaderv) * eval_batch_size)
    return loss, acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("load dataset.........................")
    # 加载数据+
    # vibration 路径
    trainpathv = r'E:\1. Manuscripts\paper_10 (Class-imbalanced Multi-source Information Fusion Transformer-based Neural Network for Mechanical Fault Diagnosis with Limited Data)\2. Ours\models\data\case3\C1\image\Vibration\train/'
    valpathv = r'E:\1. Manuscripts\paper_10 (Class-imbalanced Multi-source Information Fusion Transformer-based Neural Network for Mechanical Fault Diagnosis with Limited Data)\2. Ours\models\data\case3\C1\image\Vibration\test/'
    # acoustic 路径
    trainpatha = r'E:\1. Manuscripts\paper_10 (Class-imbalanced Multi-source Information Fusion Transformer-based Neural Network for Mechanical Fault Diagnosis with Limited Data)\2. Ours\models\data\case3\C1\image\Acoustic\train/'
    valpatha = r'E:\1. Manuscripts\paper_10 (Class-imbalanced Multi-source Information Fusion Transformer-based Neural Network for Mechanical Fault Diagnosis with Limited Data)\2. Ours\models\data\case3\C1\image\Acoustic\test/'

    img_size = 32
    train_batch_size = 8
    eval_batch_size = 4
    learning_rate = 0.002
    weight_decay = 1e-3
    total_epoch = 50

    # 加载模型
    model = MSIF(128, 3, 4, 32, 512, 0).to(device)

    # vibration 路径
    train_linesv = os.listdir(trainpathv)
    val_linesv = os.listdir(valpathv)
    train_datasetv = netDataset(trainpathv, train_linesv, img_size)
    val_datasetv = netDataset(valpathv, val_linesv, img_size)
    train_loaderv = DataLoader(train_datasetv, shuffle=True, batch_size=train_batch_size,
                               num_workers=0, pin_memory=True, drop_last=True)
    test_loaderv = DataLoader(val_datasetv, shuffle=False, batch_size=eval_batch_size,
                              num_workers=0, pin_memory=True, drop_last=True)
    # acoustic 路径
    train_linesa = os.listdir(trainpatha)
    val_linesa = os.listdir(valpatha)
    train_dataseta = netDataset(trainpatha, train_linesa, img_size)
    val_dataseta = netDataset(valpatha, val_linesa, img_size)
    train_loadera = DataLoader(train_dataseta, shuffle=True, batch_size=train_batch_size,
                               num_workers=0, pin_memory=True, drop_last=True)
    test_loadera = DataLoader(val_dataseta, shuffle=False, batch_size=eval_batch_size,
                              num_workers=0, pin_memory=True, drop_last=True)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_function_binary = torch.nn.BCEWithLogitsLoss()
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    print("training.........................")
    # 设置测试损失list,和测试acc 列表
    val_loss_list = []
    val_acc_list = []
    # 设置训练损失list
    train_loss_list = []
    train_acc_list = []
    max_acc = 0
    for i in trange(total_epoch, desc='Training', unit='epoch'):
        model.train()
        train_loss = 0
        for batch1, batch2 in zip(train_loaderv, train_loadera):
            xv, y = batch1
            xa, _ = batch2
            xv = xv.to(device)
            xa = xa.to(device)
            y = y.to(device)
            # 将多分类标签改为二分类标签
            y_0 = y
            binary_y = torch.where(y_0 > 1, 1, y_0)
            binary_y = F.one_hot(binary_y, num_classes=2)
            binary_y = binary_y.to(torch.float)

            # 定义模型
            binary_logits, logits, D_vib_logits, D_aco_logits, weight = model(xv, xa)
            # print(logits.shape)
            # discriminator 损失
            vib_gt = torch.ones([D_vib_logits.shape[0]], dtype=torch.int64).to(xv.device)
            aco_gt = torch.zeros([D_aco_logits.shape[0]], dtype=torch.int64).to(xv.device)
            ad_loss = (loss_function(D_vib_logits, vib_gt) + loss_function(D_aco_logits, aco_gt)) / 2
            # 二分类损失
            loss_binary = loss_function_binary(binary_logits, binary_y)  # binary_loss
            # 多分类损失
            loss_multi = torch.sum(weight * loss_function(logits, y))
            loss = ad_loss + 0.4 * loss_binary + 0.6 * loss_multi
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 每训练一个epoch,记录一次训练损失
        train_loss = train_loss / len(train_loaderv)
        train_loss_list.append(train_loss)
        _, train_acc = eval(model, train_loaderv, train_loadera)
        train_acc_list.append(train_acc)

        print("train Epoch:{},loss:{},train_acc:{}".format(i, train_loss, train_acc / 2))

        # 每训练一个epoch,用当前训练的模型对验证集进行测试
        eval_loss, eval_acc = eval(model, test_loaderv, test_loadera)
        # 将每一个测试集验证的结果加入列表
        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)

        print("val Epoch:{},eval_loss:{},eval_acc:{}".format(i, eval_loss, eval_acc))
        if eval_acc > max_acc:
            max_acc = eval_acc
            # 保存最优模型参数
            torch.save(model, 'output/case2/B8/best.pt')
    torch.save(model, 'output/case2/B8/last.pt')  # 保存最后一个epoch的模型
    np.savetxt("output/case2/B8/train_loss_list.txt", train_loss_list)
    np.savetxt("output/case2/B8/train_acc_list.txt", train_acc_list)
    np.savetxt("output/case2/B8/val_loss_list.txt", val_loss_list)
    np.savetxt("output/case2/B8/val_acc_list.txt", val_acc_list)

    with open('output/case2/B8/train_loss_list.txt', 'r') as f:
        train_loss_list = f.readlines()
    train_loss = [float(i.strip()) for i in train_loss_list]
    with open('output/case2/B8/val_loss_list.txt', 'r') as f:
        val_loss_list = f.readlines()
    val_loss = [float(i.strip()) for i in val_loss_list]

    with open('output/case2/B8/train_acc_list.txt', 'r') as f:
        train_acc_list = f.readlines()
    train_acc = [float(i.strip()) for i in train_acc_list]
    with open('output/case2/B8/val_acc_list.txt', 'r') as f:
        val_acc_list = f.readlines()
    val_acc = [float(i.strip()) for i in val_acc_list]

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig('result/loss curve.jpg')
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.savefig('result/accuracy curve.jpg')
    plt.show()
