import torch
import torch.nn as nn
import torch.optim as optim

from preprocess import load_data
from attention_augmented_wide_resnet import Wide_ResNet

import argparse
from tqdm import tqdm
import time

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_args():
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--dataset-mode", type=str, default="CIFAR100", help="(example: CIFAR10, CIFAR100, MNIST), (default: CIFAR100)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8, help="number of batch size, (default, 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--depth", type=int, default=28, help="wide-ResNet depth, (default: 28)")
    parser.add_argument("--widen_factor", type=int, default=10, help="wide_ResNet widen factor, (default: 10)")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate, (default: 0.3)")

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length


def get_test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    return acc


def main():
    args = get_args()
    train_loader, test_loader = load_data(args)

    if args.dataset_mode == "CIFAR10":
        num_classes = 10
    elif args.dataset_mode == "CIFAR100":
        num_classes = 100
    elif args.dataset_mode == "MNIST":
        num_classes = 10

    model = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    max_acc = 0
    for epoch in range(1, args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, args)
        test_acc = get_test(model, test_loader)
        if max_acc < test_acc:
            max_acc = test_acc
        print("Test acc:", max_acc, "time: ", time.time() - start_time)


if __name__ == "__main__":
    main()
