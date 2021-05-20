
import torch
import matplotlib.pyplot as plt
from PIL import Image



# helper custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(CustomDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# augmentation helper
class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_fn, num_labels=1):
        self.dataset = dataset
        self.map = map_fn
        self.num_labels = num_labels

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # image

        if self.num_labels == 1:
            y = self.dataset[index][1]  # label 1
            return (x, y)

        if self.num_labels == 2:
            y1 = self.dataset[index][1]  # label 1
            y2 = self.dataset[index][2]  # label 2
            return (x, y1, y2)

    def __len__(self):
        return len(self.dataset)


# image plotting helper
def show_from_tensor(tensor, title=None, save=False):
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().squeeze()
    plt.figure()
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    if save == True:
        plt.imsave('drive/My Drive/cw2/results/{}.jpg'.format(title), img, cmap='gray')
    plt.pause(0.001)


# Bland-Altman plot helper
def bland_altman(A, B):
    diff = A - B
    mean = (A + B)/2

    std = np.std(diff)
    mean_diff = np.mean(diff)

    plt.figure()
    plt.xlabel('Mean of Sampling 1 and Sampling 2')
    plt.ylabel('Sampling 1 - Sampling 2')
    plt.scatter(mean, diff, s=1)
    plt.axhline(mean_diff, label=('Mean Difference'+r'$\approx %0.2f$'%mean_diff))
    plt.axhline(mean_diff + 1.96 * std, label=('$\pm 1.96$ std' + r'$\approx %0.2f$' % 1.96*std))
    plt.axhline(mean_diff - 1.96 * std)
    plt.show()


pred_image = Image.open("/home/taymaz/Documents/mphy0041/cw2/example_results/consensus prediction for train image 50, accuracy_ 0.970.jpg")
frame_image = Image.open("/home/taymaz/Documents/mphy0041/cw2/example_results/train image 50.jpg")
label_image = Image.open("/home/taymaz/Documents/mphy0041/cw2/example_results/label for train image 50.jpg")

blended_pred_seg = Image.blend(frame_image, pred_image, alpha=.3)
blended_label = Image.blend(frame_image, label_image, alpha=.3)

blended_pred_seg.save("/home/taymaz/Documents/mphy0041/cw2/example_results/pred_overlay.png")
blended_label.save("/home/taymaz/Documents/mphy0041/cw2/example_results/truth_overlay.png")

