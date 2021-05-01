from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import random

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=15, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(15, 15, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(15, 12, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(12, 8, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)
        self.batch1 = nn.BatchNorm2d(15)
        self.batch2 = nn.BatchNorm2d(15)
        self.batch3 = nn.BatchNorm2d(12)
        self.batch4 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(1568, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = F.avg_pool2d(x, 2)
        #x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout3(x)

        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        #print("l;sadf;jlksjkl;fa;")
        x = F.relu(x)
        # print(x.shape)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def only_conv(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = F.avg_pool2d(x, 2)
        #x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        #x = self.dropout3(x)

        x = torch.flatten(x, 1)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    tot_loss = 0.0
    correct = 0
    train_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_num += len(data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))
        tot_loss += loss / len(target)
    tot_loss /= len(train_loader)
    return tot_loss.detach().cpu().numpy(), 100. * correct / train_num


def test(model, device, test_loader, show_wrong=False, c_mat=True, tsne=False):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    incorrect = []
    preds = []
    gt = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

            if c_mat:
                preds = preds + list(pred.cpu().numpy())
                gt = gt + list(target.detach().cpu().numpy())

            if show_wrong:
                for i, label in enumerate(target):
                    if label != pred[i]:
                        incorrect.append([data[i][0].cpu().numpy(), label, pred[i].cpu().numpy()[0]])

    test_loss /= test_num

    if c_mat:
        print("confusion matrix:\n", confusion_matrix(gt, preds))

    if tsne:
        X = []
        images = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model.only_conv(data)
                gt = gt + list(target.detach().cpu().numpy())
                for image in list(data.detach().cpu().numpy()):
                    images.append(image[0])
                for feature_map in output:
                    X.append(feature_map.cpu().numpy())
        X_embedded = TSNE(n_components=2).fit_transform(X)
        print(X_embedded.shape)
        plt.figure()
        #, label = str(gt)
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=gt, alpha=.5)
        plt.title("TSNE Visualization for Feature Map")
        plt.legend(*scatter.legend_elements(), title="Number")
        plt.axis('off')
        plt.show()

        n = 4
        m = 8
        plt.figure()
        for i in range(n):
            I = [np.array(random.choice(np.arange(len(X_embedded))))]
            dists = [[np.linalg.norm(X_embedded[I[0]] - x), i] for i, x in enumerate(X_embedded)]
            dists.sort(key=lambda x: x[0])
            for k in range(m):
                I.append(dists[k+1][1])

            ims = []
            for index in I:
                ims.append(images[index])

            #print(ims[0])

            for j in range(0, m+1):
                plt.subplot(n, m+1, i*(m+1)+(j+1))
                plt.imshow(ims[j])
                plt.axis('off')
        plt.show()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    if show_wrong:
        plt.figure()
        for i, (img, label, pred) in enumerate(incorrect):
            plt.subplot(331 + i)
            plt.axis('off')
            plt.title(f'Label: {label}, Pred: {pred}')
            plt.imshow(img)
            if i == 8:
                break
        plt.show()

    return test_loss, 100. * correct / test_num


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--traintest', action='store_true', default=False,
                        help='train on fractions of data and test')
    parser.add_argument('--show_wrong', action='store_true', default=False,
                        help='show incorrectly labelled test images')
    parser.add_argument('--show_features', action='store_true', default=False,
                        help='show first layer features')
    parser.add_argument('--show_tsne', action='store_true', default=False,
                        help='show first layer features')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # show features
    if args.show_features:
        assert os.path.exists(args.load_model)

        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        conv1 = model.conv1.weight.detach().cpu()

        plt.figure()
        for i, filt in enumerate(conv1):
            plt.subplot(331 + i)
            plt.axis('off')
            plt.title(f'Filter {i+1}')
            plt.imshow(filt[0])
            if i == 8:
                break
        plt.show()
        return


    # Evaluate on the official test set
    if args.evaluate or args.show_wrong:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader, show_wrong=args.show_wrong, tsne=args.show_tsne)

        return


    if args.traintest:
        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),
                                                               (0.3081,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True,
            **kwargs)

    # Pytorch has default MNIST dataloader which loads data at each iteration
    data_transforms = {"train": transforms.Compose([
                            transforms.RandomRotation(degrees=10),
                            #transforms.RandomAffine(degrees=10, scale=(.9, .9)),# scale=(.1, .1), shear=.1),
                            transforms.ToTensor(),
                            #transforms.RandomErasing(p=0.8, scale=(.05, .05)),
                            #transforms.Normalize((0.1307,), (0.3081,))
                            ]),
                        "val": transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])
                    }
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=data_transforms["train"])
    val_dataset = datasets.MNIST('../data', train=False, download=True, transform=data_transforms["val"])

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.

    class_data = [[] for _ in range(10)]

    for i, elem in enumerate(tqdm(val_dataset)):
        class_data[elem[1]].append(i)

    split = .85
    subset_indices_train = []
    subset_indices_valid = []
    for i in range(10):
        np.random.shuffle(class_data[i])
        subset_indices_train += class_data[i][:int(len(class_data[i])*split)]
        subset_indices_valid += class_data[i][int(len(class_data[i])*split):]

    if args.traintest:
        train_fracs = [1., .5, .25, .125, .0625]
    else:
        train_fracs = [1.]

    train_frac_losses = []
    test_frac_losses = []
    train_frac_accs = []
    test_frac_accs = []
    for train_frac in train_fracs:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train[:int(len(subset_indices_train)*train_frac)])
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.test_batch_size,
            sampler=SubsetRandomSampler(subset_indices_valid[:int(len(subset_indices_valid)*train_frac)])
        )

        #for imgs, label in train_loader:
        #    for img in imgs:
        #        plt.figure()
        #        plt.imshow(img[0])
        #        plt.show()

        # Load your model [fcNet, ConvNet, Net]
        model = Net().to(device)

        # Try different optimzers here [Adam, SGD, RMSprop]
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # Set your learning rate scheduler
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

        # Training loop
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)
            test_loss, test_acc = test(model, device, val_loader)
            scheduler.step()    # learning rate scheduler
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            # You may optionally save your model at each epoch here

        if not args.traintest:
            figs, axs = plt.subplots(2)
            axs[0].set_title("Loss")
            axs[0].set_ylabel("Loss")
            axs[0].plot(test_losses)
            axs[0].plot(train_losses)
            axs[1].set_title("Accuracy")
            axs[1].set_ylabel("Accuracy")
            axs[1].plot(test_accs)
            axs[1].set_xlabel("Epoch")
            plt.show()
        else:
            test_loss, test_acc = test(model, device, test_loader)
            test_frac_losses.append(test_loss)
            train_frac_losses.append(train_loss)
            test_frac_accs.append(test_acc)
            train_frac_accs.append(train_acc)

    if args.traintest:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Loss")
        plt.plot(train_fracs, test_frac_losses)
        plt.plot(train_fracs, train_frac_losses)
        plt.legend(["Test", "Train"])
        plt.ylabel("Loss (log)")
        plt.yscale("log")
        plt.xscale("log")

        plt.subplot(2, 1, 2)
        plt.title("Accuracy")
        plt.plot(train_fracs, test_frac_accs)
        plt.plot(train_fracs, train_frac_accs)
        plt.legend(["Test", "Train"])
        plt.xlabel("Fraction of Training Set (log)")
        plt.ylabel("Accuracy (log)")
        plt.yscale("log")
        plt.xscale("log")
        plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
