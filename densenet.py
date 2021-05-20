import torch
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from dataloader import *
from utils import *


#################### MODEL ######################

def buildDenseNet(numClasses):
    # get the stock PyTorch DenseNet model
    model = models.densenet161()

    # change 1st conv layer from 3 channel to 1 channel
    model.features.conv0 = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # number of input features to the final layer
    numInputs = model.classifier.in_features
    # replace the final layer with custom number of classes and softmax
    model.classifier = nn.Sequential(
        nn.Linear(in_features=numInputs, out_features=numClasses, bias=True),
        nn.Softmax(dim=1))

    return model


#################### TRAINER ######################

def train_densenet(args):

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # load dataset
    dataset = H5Dataset_consensus(file_path='drive/My Drive/cw2/dataset70-200.h5')
    # load dataset used for bootstrap evaluation - select random seed to keep consistency between bootstraps
    dataset_seed = H5Dataset_consensus(file_path='drive/My Drive/cw2/dataset70-200.h5', seed=9)

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [int(len(dataset)*args.split[0]),
                  int(len(dataset)*args.split[1]),
                  int(len(dataset)*args.split[2])])
    ensemble_train_set,  ensemble_val_set,  ensemble_test_set = torch.utils.data.random_split(
        dataset_seed, [int(len(dataset_seed)*args.split[0]),
                       int(len(dataset_seed)*args.split[1]),
                       int(len(dataset_seed)*args.split[2])])

    # define augmentation (only training data is augmented)
    H, W = dataset.dim()
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([int(1.75 * H), int(1.75 * W)]),
        transforms.RandomCrop([H,W]),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.ToTensor()])
    no_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])

    # apply augmentation
    train_set, ensemble_train_set = MapDataset(train_set, train_transform, num_labels=2), MapDataset(ensemble_train_set, no_transform, num_labels=2)
    test_set, ensemble_val_set = MapDataset(test_set, no_transform, num_labels=2), MapDataset(ensemble_val_set, no_transform, num_labels=2)
    val_set, ensemble_test_set = MapDataset(val_set, no_transform, num_labels=2), MapDataset(ensemble_test_set, no_transform, num_labels=2)

    # create training dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

    # bootstrap dataloaders need index consistency across bootstraps (unshuffled)
    ensemble_train_loader = torch.utils.data.DataLoader(ensemble_train_set, shuffle=False)
    ensemble_val_loader = torch.utils.data.DataLoader(ensemble_val_set, shuffle=False)
    ensemble_test_loader = torch.utils.data.DataLoader(ensemble_test_set, shuffle=False)

    # to allow cycle between training and validation
    data_loaders = {'train': train_loader, 'val': val_loader}

    # store predictions from each bootstrap
    ensemble_train_preds = []
    ensemble_val_preds = []
    ensemble_test_preds = []

    # bootstrap loop (outer loop)
    for bootstrap in range(1, args.bootstraps + 1):

        # make new instance of model for each bootstrap
        net = buildDenseNet(numClasses=2).to(device)

        # training loop (inner loop)
        for epoch in range(1, args.epochs + 1):

            # each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.beta)
                    net.train(True)   # set model to training mode
                else:
                    net.train(False)  # set model to validation mode

                # mini batch training
                running_loss = 0
                running_accuracy = 0

                for batch, (images, _, classes) in enumerate(data_loaders[phase]):
                    optimizer.zero_grad()

                    # send tensors to CUDA
                    images = images.to(device)
                    classes = classes.to(device)

                    # forward propagation
                    preds = net(images)
                    bce_loss = nn.BCELoss()
                    loss = bce_loss(preds, classes)

                    # optimization step (only in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # compute accuracy
                    accuracy = (preds.round() == classes).sum()/(2 * float(preds.shape[0]))

                    # print progress
                    running_loss += loss.item()
                    running_accuracy += accuracy.item()
                    if batch % args.print_interval == args.print_interval - 1:
                        print('bootstrap: {}, epoch: {}, [{}/{} ({:.0f}%)], {} loss: {:.3f}, {} accuracy:{:.3f}'.format(
                            bootstrap,
                            epoch,
                            batch * args.batch_size,
                            len(train_loader.dataset),
                            100. * batch / len(train_loader),
                            phase,
                            running_loss / args.print_interval,
                            phase,
                            running_accuracy / args.print_interval))
                        running_loss = 0
                        running_accuracy = 0

        # make and store predictions from the fully trained bootstrap
        net.train(False)
        for train_image, _, _ in ensemble_train_loader:
            train_pred = net(train_image.to(device))
            ensemble_train_preds.append(train_pred.cpu().detach())
        for val_image, _, _ in ensemble_val_loader:
            val_pred = net(val_image.to(device))
            ensemble_val_preds.append(val_pred.cpu().detach())
        for test_image, _, _ in ensemble_test_loader:
            test_pred = net(test_image.to(device))
            ensemble_test_preds.append(test_pred.cpu().detach())

    # collect predictions from all bootstraps
    ensemble_train_preds = torch.stack(ensemble_train_preds)
    ensemble_val_preds = torch.stack(ensemble_val_preds)
    ensemble_test_preds = torch.stack(ensemble_test_preds)

    # consensus vote
    # reshape predictions to [len_data, num_bootstraps, 1, num_classes]
    ensemble_train_preds = torch.reshape(ensemble_train_preds, (len(ensemble_train_loader), args.bootstraps, 1, 2))
    ensemble_val_preds = torch.reshape(ensemble_val_preds, (len(ensemble_val_loader), args.bootstraps, 1, 2))
    ensemble_test_preds = torch.reshape(ensemble_test_preds, (len(ensemble_test_loader), args.bootstraps, 1, 2))

    # take average class prediction across bootstraps
    mean_train_preds = torch.squeeze(torch.sum(ensemble_train_preds.round(), dim=1) / args.bootstraps)
    mean_val_preds = torch.squeeze(torch.sum(ensemble_val_preds.round(), dim=1) / args.bootstraps)
    mean_test_preds = torch.squeeze(torch.sum(ensemble_test_preds.round(), dim=1) / args.bootstraps)

    # relabel classes according to 0.5 threshold
    consensus_train_pred = []
    for mean_pred in mean_train_preds:
        if mean_pred[0] > 0.5:
            consensus_train_pred.append(torch.tensor([1, 0]))
        else:
            consensus_train_pred.append(torch.tensor([0, 1]))

    consensus_val_pred = []
    for mean_pred in mean_val_preds:
        if mean_pred[0] > 0.5:
            consensus_val_pred.append(torch.tensor([1, 0]))
        else:
            consensus_val_pred.append(torch.tensor([0, 1]))

    consensus_test_pred = []
    for mean_pred in mean_test_preds:
        if mean_pred[0] > 0.5:
            consensus_test_pred.append(torch.tensor([1, 0]))
        else:
            consensus_test_pred.append(torch.tensor([0, 1]))

    consensus_train_pred = torch.stack(consensus_train_pred)
    consensus_test_pred = torch.stack(consensus_test_pred)

    # remove data which had no detection
    screened_train_images = []
    screened_train_labels = []
    for ((image, label, _), pred) in zip(ensemble_train_loader, consensus_train_pred):
        if pred[0] == 1:   # corresponds to [1,0]
            screened_train_images.append(torch.squeeze(image, dim=0).cpu())
            screened_train_labels.append(torch.squeeze(label, dim=0).cpu())
    screened_train_set = CustomDataset(torch.stack(screened_train_images), torch.stack(screened_train_labels))

    screened_val_images = []
    screened_val_labels = []
    for ((image, label, _), pred) in zip(ensemble_val_loader, consensus_val_pred):
        if pred[0] == 1:   # corresponds to [1,0]
            screened_val_images.append(torch.squeeze(image, dim=0).cpu())
            screened_val_labels.append(torch.squeeze(label, dim=0).cpu())
    screened_val_set = CustomDataset(torch.stack(screened_val_images), torch.stack(screened_val_labels))

    screened_test_images = []
    screened_test_labels = []
    for ((image, label, _), pred) in zip(ensemble_test_loader, consensus_test_pred):
        if pred[0] == 1:   # corresponds to [1,0]
            screened_test_images.append(torch.squeeze(image, dim=0).cpu())
            screened_test_labels.append(torch.squeeze(label, dim=0).cpu())
    screened_test_set = CustomDataset(torch.stack(screened_test_images), torch.stack(screened_test_labels))

    return screened_train_set, screened_val_set, screened_test_set



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="use CUDA acceleration", default=True, action='store_false')
    parser.add_argument("--split", help="train, validation, test split e.g. [0.6, 0.2, 0.2]", type=list, default=[0.6, 0.2, 0.2])
    parser.add_argument("--batch_size", help="train batch size", type=int, default=16)
    parser.add_argument("--bootstraps", help="number of re-trainings", type=int, default=1)
    parser.add_argument("--epochs", help="train epochs", type=int, default=100)
    parser.add_argument("--lr", help="adam learning rate", type=int, default=1e-4)
    parser.add_argument("--beta", help="L2 regularization", type=int, default=0)
    parser.add_argument("--print_interval", help="interval to print batch loss", type=int, default=2)
    args = parser.parse_args()

    train_densenet(args)