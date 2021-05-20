
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from dataloader import *
from utils import *


#################### MODEL ######################

class ResBlock(nn.Module):
    def __init__(self, num_ch, stride=3, dropout=0, expansion=4, zero_initialization=True):
        super(ResBlock, self).__init__()

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(num_ch, num_ch*expansion, kernel_size=stride, padding=1),
            nn.Conv2d(num_ch*expansion, num_ch, kernel_size=stride, padding=1)])
        self.batch_norm_layers = nn.ModuleList(
            [nn.BatchNorm2d(num_features=num_ch*expansion),
            nn.BatchNorm2d(num_features=num_ch)])
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        if zero_initialization:
            torch.nn.init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        temps = self.conv_layers[0](temps)
        temps = self.batch_norm_layers[0](temps)
        temps = self.relu(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        temps = self.batch_norm_layers[1](temps)
        temps = self.relu(temps)
        return inputs + temps


class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch, stride=3, num_blocks=4, dropout=0):
        super().__init__()

        self.conv_initial = nn.Conv2d(in_ch, mid_ch, kernel_size=stride, padding=1)
        self.conv_blocks = nn.ModuleList(
            [ResBlock(mid_ch, stride=stride, dropout=dropout,) for _ in range(num_blocks)])
        self.conv_final = nn.Conv2d(mid_ch, out_ch, kernel_size=stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        temps = self.conv_initial(inputs)
        temps = self.relu(temps)
        for block in self.conv_blocks:
            temps = block(temps)
        outputs = self.conv_final(temps)
        return outputs


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, features=[64,128,256]):
        super(UNet, self).__init__()

        self.down_conv = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder
        for feature in features:
            self.down_conv.append(ResNet(in_ch, feature, feature))
            in_ch = feature

        # decoder
        for feature in reversed(features):
            self.up_trans.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.up_trans.append(ResNet(feature*2, feature, feature))

        self.bottleneck = ResNet(features[-1], features[-1]*2, features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.down_conv:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_trans), 2):
            x = self.up_trans[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = tv.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_trans[idx+1](concat_skip)
        x = self.final_conv(x)
        return torch.sigmoid(x)


#################### METRICS ######################

def loss_dice(y_pred, y_true, eps=1e-6, dim=(2,3)):
    numerator = 2 * torch.sum(y_true*y_pred, dim=dim)
    denominator = torch.sum(y_true, dim=dim) + torch.sum(y_pred, dim=dim) + eps
    return torch.mean(1. - (numerator / denominator))


def jaccards_index(y_pred, y_true, dim=(2,3)):
    numerator = torch.sum(y_true*y_pred, dim=dim)
    denominator = torch.sum(y_true, dim=dim) + torch.sum(y_pred, dim=dim) - numerator
    return torch.mean(numerator / denominator)


#################### TRAINER ######################

def train_unet(args):

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # load dataset
    dataset = H5Dataset_random(file_path='drive/My Drive/cw2/dataset70-200.h5')
    # load dataset used for bootstrap evaluation - select random seed to keep consistency between bootstraps
    dataset_seed = H5Dataset_random(file_path='drive/My Drive/cw2/dataset70-200.h5', seed=9)

    # train, val, test split
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
        transforms.Resize([int(1.25 * H), int(1.25 * W)]),
        transforms.RandomCrop([H,W]),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.ToTensor()])
    no_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])

    # apply augmentation
    train_set, ensemble_train_set = MapDataset(train_set, train_transform), MapDataset(ensemble_train_set, no_transform)
    test_set, ensemble_val_set = MapDataset(test_set, no_transform), MapDataset(ensemble_val_set, no_transform)
    val_set, ensemble_test_set = MapDataset(val_set, no_transform), MapDataset(ensemble_test_set, no_transform)

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
        net = UNet(in_ch=1, out_ch=1, features=args.features).to(device)

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

                for batch, (images, labels) in enumerate(data_loaders[phase]):
                    optimizer.zero_grad()

                    # send tensors to CUDA
                    images = images.to(device)
                    labels = labels.to(device)

                    # forward propagation
                    preds = net(images)
                    loss = loss_dice(preds, labels)

                    # optimization step (only in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # compute accuracy metric
                    accuracy = jaccards_index(preds, labels)

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
        for train_image, train_label in ensemble_train_loader:
            train_pred = net(train_image.to(device))
            ensemble_train_preds.append(train_pred.cpu().detach())
        for val_image, val_label in ensemble_val_loader:
            val_pred = net(val_image.to(device))
            ensemble_val_preds.append(val_pred.cpu().detach())
        for test_image, test_label in ensemble_test_loader:
            test_pred = net(test_image.to(device))
            ensemble_test_preds.append(test_pred.cpu().detach())

    # collect predictions from all bootstraps
    ensemble_train_preds = torch.stack(ensemble_train_preds)
    ensemble_val_preds = torch.stack(ensemble_val_preds)
    ensemble_test_preds = torch.stack(ensemble_test_preds)

    # consensus vote
    # reshape predictions to [len_data, num_bootstraps, H, W]
    ensemble_train_preds = torch.reshape(ensemble_train_preds, (len(ensemble_train_loader), args.bootstraps, H, W))
    ensemble_val_preds = torch.reshape(ensemble_val_preds, (len(ensemble_val_loader), args.bootstraps, H, W))
    ensemble_test_preds = torch.reshape(ensemble_test_preds, (len(ensemble_test_loader), args.bootstraps, H, W))

    # take average across bootstraps and select pixels above 0.5 threshold
    consensus_train_pred = (torch.sum(ensemble_train_preds, dim=1) > (0.5 * args.bootstraps)).long()
    consensus_val_pred = (torch.sum(ensemble_val_preds, dim=1) > (0.5 * args.bootstraps)).long()
    consensus_test_pred = (torch.sum(ensemble_test_preds, dim=1) > (0.5 * args.bootstraps)).long()

    # plot and save consensus predictions
    for i, ((image, label), pred) in enumerate(zip(ensemble_train_loader, consensus_train_pred)):
        train_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, train_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)

    for i, ((image, label), pred) in enumerate(zip(ensemble_val_loader, consensus_val_pred)):
        val_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, val_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)

    for i, ((image, label), pred) in enumerate(zip(ensemble_test_loader, consensus_test_pred)):
        test_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, test_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="use CUDA acceleration", default=True, action='store_false')
    parser.add_argument("--split", help="train, validation, test split e.g. [0.6, 0.2, 0.2]", type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument("--batch_size", help="train batch size", type=int, default=16)
    parser.add_argument("--features", help="number of channels in UNet architecture e.g. [64, 128, 256]", type=list, default=[32, 64, 128])
    parser.add_argument("--bootstraps", help="number of re-trainings", type=int, default=1)
    parser.add_argument("--epochs", help="train epochs", type=int, default=10)
    parser.add_argument("--lr", help="adam learning rate", type=int, default=1e-5)
    parser.add_argument("--beta", help="L2 regularization", type=int, default=0)
    parser.add_argument("--print_interval", help="interval to print batch loss", type=int, default=5)
    args = parser.parse_args()

    train_unet(args)