

from densenet import *
from unet import *



def train_combine(args):

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # train classification model
    screened_train_set, screened_val_set, screened_test_set = train_densenet(args)
    torch.cuda.empty_cache()
    print('Classifier training finished')

    # create dataloaders (note train data is already augmented)
    train_loader = torch.utils.data.DataLoader(screened_train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(screened_val_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(screened_test_set, shuffle=False)

    # dataloaders for evaluation and plotting
    eval_train_loader = torch.utils.data.DataLoader(screened_train_set, shuffle=False)
    eval_val_loader = torch.utils.data.DataLoader(screened_val_set, shuffle=False)
    eval_test_loader = torch.utils.data.DataLoader(screened_test_set, shuffle=False)

    # to allow cycle between training and validation
    data_loaders = {'train': train_loader, 'val': val_loader}

    # store predictions from each bootstrap
    ensemble_train_preds = []
    ensemble_val_preds = []
    ensemble_test_preds = []

    # bootstrap loop (outer loop)
    for bootstrap in range(1, args.bootstraps + 1):
        torch.cuda.empty_cache()

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
        for train_image, train_label in eval_train_loader:
            train_pred = net(train_image.to(device))
            ensemble_train_preds.append(train_pred.cpu().detach())
        for val_image, val_label in eval_val_loader:
            val_pred = net(val_image.to(device))
            ensemble_val_preds.append(val_pred.cpu().detach())
        for test_image, test_label in eval_test_loader:
            test_pred = net(test_image.to(device))
            ensemble_test_preds.append(test_pred.cpu().detach())

    # collect predictions from all bootstraps
    ensemble_train_preds = torch.stack(ensemble_train_preds)
    ensemble_val_preds = torch.stack(ensemble_val_preds)
    ensemble_test_preds = torch.stack(ensemble_test_preds)

    # consensus vote
    H, W = 58, 52
    # reshape predictions to [len_data, num_bootstraps, H, W]
    ensemble_train_preds = torch.reshape(ensemble_train_preds, (len(eval_train_loader.dataset), args.bootstraps, H, W))
    ensemble_val_preds = torch.reshape(ensemble_val_preds, (len(eval_val_loader.dataset), args.bootstraps, H, W))
    ensemble_test_preds = torch.reshape(ensemble_test_preds, (len(eval_test_loader.dataset), args.bootstraps, H, W))

    # take average across bootstraps and select pixels above 0.5 threshold
    consensus_train_pred = (torch.sum(ensemble_train_preds, dim=1) > (0.5 * args.bootstraps)).long()
    consensus_val_pred = (torch.sum(ensemble_val_preds, dim=1) > (0.5 * args.bootstraps)).long()
    consensus_test_pred = (torch.sum(ensemble_test_preds, dim=1) > (0.5 * args.bootstraps)).long()

    # plot and save consensus predictions
    for i, ((image, label), pred) in enumerate(zip(eval_train_loader, consensus_train_pred)):
        train_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, train_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)

    for i, ((image, label), pred) in enumerate(zip(eval_val_loader, consensus_val_pred)):
        val_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, val_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)

    for i, ((image, label), pred) in enumerate(zip(eval_test_loader, consensus_test_pred)):
        test_accuracy = jaccards_index(pred.cpu(), torch.squeeze(label).cpu(), dim=(0,1))
        show_from_tensor(image, title='train image {}'.format(i), save=True)
        show_from_tensor(pred, title='consensus prediction for train image {}, accuracy: {:.3f}'.format(i, test_accuracy), save=True)
        show_from_tensor(label, title='label for train image {}'.format(i), save=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="use CUDA acceleration", default=True, action='store_false')
    parser.add_argument("--split", help="train, validation, test split e.g. [0.6, 0.2, 0.2]", type=list,
                        default=[0.6, 0.2, 0.2])
    parser.add_argument("--batch_size", help="train batch size", type=int, default=16)
    parser.add_argument("--features", help="number of channels in UNet architecture e.g. [64, 128, 256]", type=list,
                        default=[32, 64, 128, 256])
    parser.add_argument("--bootstraps", help="number of re-trainings", type=int, default=1)
    parser.add_argument("--epochs", help="train epochs", type=int, default=100)
    parser.add_argument("--lr", help="adam learning rate", type=int, default=1e-4)
    parser.add_argument("--beta", help="L2 regularization", type=int, default=0)
    parser.add_argument("--print_interval", help="interval to print batch loss", type=int, default=2)
    args = parser.parse_args()

    train_combine(args)