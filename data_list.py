#from __future__ import print_function, division

import scipy.stats
import numpy as np
import os
import os.path
import pickle
import random
import sys
import time
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms


def write_list(f, l):
    f.write(",".join(map(str, l)) + "\n")
    f.flush()


def sample_ratios(samples, labels, ratios=None):
    if ratios is not None:
        selected_idx = []
        for i, ratio in enumerate(ratios):
            images_i = [j for j in range(
                labels.shape[0]) if labels[j].item() == i]
            num_ = len(images_i)
            idx = np.random.choice(
                num_, int(ratio * num_), replace=False)
            selected_idx.extend(np.array(images_i)[idx].tolist())
        return samples[selected_idx, :, :], labels[selected_idx]

    return samples, labels


def image_classification_test_loaded(test_samples, test_labels, model, test_10crop=True, device='cpu'):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        if test_10crop:
            len_test = test_labels[0].shape[0]
            for i in range(len_test):
                outputs = []
                for j in range(10):
                    data, target = test_samples[j][i].unsqueeze(
                        0), test_labels[j][i].unsqueeze(0)
                    _, output = model(data)
                    test_loss += nn.CrossEntropyLoss()(output, target).item()
                    outputs.append(nn.Softmax(dim=1)(output))
                outputs = sum(outputs)
                pred = torch.max(outputs, 1)[1]
                correct += pred.eq(target.data.cpu().view_as(pred)
                                   ).sum().item()
        else:
            len_test = test_labels.shape[0]
            bs = 72
            for i in range(int(len_test / bs)):
                data, target = torch.Tensor(
                    test_samples[bs*i:bs*(i+1)]).to(device), test_labels[bs*i:bs*(i+1)]
                output = model(data)
                test_loss += nn.CrossEntropyLoss()(output, target).item()
                pred = torch.max(output, 1)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
            # Last test samples
            data, target = torch.Tensor(
                test_samples[bs*(i+1):]).to(device), test_labels[bs*(i+1):]
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    accuracy = correct / len_test
    test_loss /= len_test
    return accuracy


def make_dataset(image_list, labels, ratios=None):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    if ratios:
        selected_images = []
        for i, ratio in enumerate(ratios):
            images_i = [img for img in images if img[1] == i]
            num_ = len(images_i)
            idx = np.random.choice(num_, int(ratio * num_), replace=False)
            for j in idx:
                selected_images.append(images_i[j])
        return selected_images

    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def build_uspsmnist(l, path, root_folder, device='cpu'):
    dset_source = ImageList(open(l).readlines(), transform=transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]), mode='L', root_folder=root_folder)
    loaded_dset_source = LoadedImageList(dset_source)
    with open(path, 'wb') as f:
        pickle.dump([loaded_dset_source.samples.numpy(),
                     loaded_dset_source.targets.numpy()], f)
    return loaded_dset_source.samples.to(device
                                         ), loaded_dset_source.targets.to(device)


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB', root_folder='', ratios=None):
        imgs = make_dataset(image_list, labels, ratios=ratios)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root_folder + "\n"))

        self.root_folder = root_folder
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root_folder, path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class LoadedImageList(Dataset):
    def __init__(self, image_list):

        self.image_list = image_list
        self.samples, self.targets = self._load_imgs()

    def _load_imgs(self):

        loaded_images, targets = [], []
        t = time.time()
        print("{} samples to process".format(len(self.image_list.imgs)))
        for i, (path, target) in enumerate(self.image_list.imgs):
            if i % 1000 == 999:
                print("{} samples in {} seconds".format(i, time.time()- t))
                sys.stdout.flush()
            img = self.image_list.loader(os.path.join(self.image_list.root_folder, path))
            if self.image_list.transform is not None:
                img = self.image_list.transform(img)
            if self.image_list.target_transform is not None:
                target = self.image_list.target_transform(target)
            loaded_images.append(img)
            targets.append(target)

        return torch.stack(loaded_images), torch.LongTensor(targets)

    def __len__(self):
        return len(self.image_list.imgs)


# List of fractions used to produce the dataset in the Performance vs J_SD paragraph. Figures 1 and 2.
# 50 elements, representing the fractions, and the theoretical JSD they would produce.
subsampling = [[[0.4052, 0.2398, 0.7001, 0.7178, 0.2132, 0.4887, 0.9849, 0.814, 0.8186, 0.2134], 0.032760409197316966],
                [[0.7872, 0.6374, 0.8413, 0.3612, 0.4427, 0.5154, 0.8423, 0.6748, 0.1634, 0.4098], 0.021267958321062656],
                [[0.2789, 0.5613, 0.9585, 0.2165, 0.9446, 0.869, 0.838, 0.9575, 0.1829, 0.7934], 0.03455597584383796],
                [[0.5017, 0.4318, 0.6189, 0.1269, 0.8788, 0.4277, 0.3875, 0.5378, 0.5405, 0.8286], 0.022225051878141944],
                [[0.2642, 0.1912, 0.1133, 0.7818, 0.9773, 0.9555, 0.741, 0.6051, 0.2096, 0.8654], 0.05013218434133124],
                [[0.2005, 0.579, 0.5852, 0.814, 0.3936, 0.6541, 0.6803, 0.3873, 0.5198, 0.7423], 0.01487691279068412],
                [[0.6315, 0.9692, 0.5519, 0.317, 0.662, 0.9937, 0.577, 0.895, 0.2807, 0.5573], 0.01786388435441286],
                [[0.6593, 0.1823, 0.9087, 0.323, 0.117, 0.324, 0.9852, 0.8075, 0.9799, 0.1527], 0.05721295477923642],
                [[0.9851, 0.5631, 0.3203, 0.227, 0.5126, 0.3978, 0.5219, 0.9644, 0.427, 0.8089], 0.02346584399291991],
                [[0.9434, 0.4466, 0.8503, 0.7235, 0.405, 0.8225, 0.2699, 0.6799, 0.5503, 0.8828], 0.01554582865827399],
                [[0.7742, 0.9749, 0.895, 0.3821, 0.7929, 0.5722, 0.5327, 0.4774, 0.7049, 0.1087], 0.026333395357855234],
                [[0.474, 0.4977, 0.4882, 0.7236, 0.4274, 0.505, 0.1939, 0.6071, 0.7333, 0.6566], 0.012314910203020654],
                [[0.7469, 0.1428, 0.7743, 0.261, 0.509, 0.6714, 0.3558, 0.2003, 0.2123, 0.7144], 0.03758103804746866],
                [[0.6078, 0.9925, 0.6132, 0.5107, 0.7755, 0.266, 0.9181, 0.5503, 0.9753, 0.787], 0.014173062912087135],
                [[0.4534, 0.3197, 0.641, 0.4995, 0.8052, 0.5237, 0.6757, 0.6292, 0.8155, 0.8259], 0.009193941950225593],
                [[0.3339, 0.7985, 0.3739, 0.8049, 0.4443, 0.5783, 0.6133, 0.7278, 0.9127, 0.7342], 0.011946408195133998],
                [[0.513, 0.6891, 0.8503, 0.7699, 0.8338, 0.8842, 0.2925, 0.5097, 0.8512, 0.9673], 0.011943243518739478],
                [[0.2608, 0.6524, 0.8783, 0.5355, 0.8859, 0.325, 0.8223, 0.2732, 0.3384, 0.733], 0.024573748979216877],
                [[0.4179, 0.8885, 0.778, 0.5943, 0.4063, 0.7704, 0.3782, 0.3344, 0.784, 0.7756], 0.01417171679861576],
                [[0.721, 0.4303, 0.5113, 0.8499, 0.6216, 0.3463, 0.7363, 0.438, 0.2705, 0.6982], 0.013848046051355988],
                [[0.9136, 0.964, 0.9482, 0.7971, 0.4281, 0.4715, 0.8894, 0.272, 0.3951, 0.7778], 0.019194643091215248],
                [[0.4694, 0.5974, 0.6888, 0.7073, 0.5244, 0.5828, 0.4859, 0.8798, 0.6837, 0.3665], 0.006838157001020787],
                [[0.5397, 0.8727, 0.3284, 0.781, 0.3955, 0.88, 0.3357, 0.8478, 0.8832, 0.4495], 0.017877156258920744],
                [[0.4096, 0.6769, 0.3566, 0.3863, 0.6231, 0.7828, 0.5524, 0.5988, 0.8546, 0.3649], 0.011476193912275862],
                [[0.6329, 0.5326, 0.8094, 0.556, 0.2853, 0.2693, 0.4511, 0.6068, 0.4359, 0.7832], 0.014025816871972494],
                [[0.6955, 0.9107, 0.9266, 0.7358, 0.7268, 0.5914, 0.7524, 0.9486, 0.6223, 0.92], 0.003294516531428644],
                [[0.6221, 0.5749, 0.4356, 0.2805, 0.3879, 0.8183, 0.9777, 0.3194, 0.6435, 0.7227], 0.017521988550845864],
                [[0.7197, 0.4776, 0.9488, 0.9869, 0.6423, 0.3082, 0.4404, 0.7307, 0.9051, 0.5027], 0.014615851037701265],
                [[0.7597, 0.9354, 0.6758, 0.9101, 0.8474, 0.6598, 0.7983, 0.6623, 0.8925, 0.7054], 0.0020994981073005157],
                [[0.8551, 0.6381, 0.6828, 0.8635, 0.6717, 0.6722, 0.8073, 0.7905, 0.9169, 0.7565], 0.0017818497157099332],
                [[0.7233, 0.7575, 0.8205, 0.6374, 0.6596, 0.7792, 0.8736, 0.8413, 0.807, 0.7333], 0.0011553039532261685],
                [[0.8778, 0.9661, 0.6376, 0.9134, 0.8296, 0.7027, 0.977, 0.8485, 0.6357, 0.7192], 0.0029014496151538102],
                [[0.9774, 0.9047, 0.9077, 0.9301, 0.9926, 0.93, 0.9202, 0.9849, 0.9231, 0.9612], 0.0001353783157801754],
                [[0.6304, 0.1495, 0.8439, 0.3944, 0.8693, 0.1466, 0.5952, 0.5705, 0.5734, 0.9339], 0.0337553984425215],
                [[0.6239, 0.9361, 0.7968, 0.7829, 0.7501, 0.1326, 0.6832, 0.1769, 0.616, 0.319], 0.033912799130997845],
                [[0.1608, 0.1367, 0.797, 0.655, 0.3387, 0.539, 0.4643, 0.6889, 0.3485, 0.1786], 0.037503662696288936],
                [[0.7596, 0.6143, 0.1171, 0.4069, 0.5707, 0.2775, 0.6583, 0.1072, 0.9349, 0.3256], 0.044312727351805296],
                [[0.1518, 0.9251, 0.2583, 0.5876, 0.4173, 0.9602, 0.1169, 0.5076, 0.9346, 0.8579], 0.04662314442412397],
                [[0.1608, 0.9229, 0.3684, 0.9401, 0.2972, 0.8121, 0.332, 0.1203, 0.8746, 0.6694], 0.046668910811104636],
                [[0.9124, 0.7087, 0.5254, 0.3416, 0.6167, 0.243, 0.3906, 0.1258, 0.1402, 0.749], 0.04128706833470013],
                [[0.4195, 0.778, 0.2067, 0.7385, 0.2988, 0.1888, 0.1895, 0.2513, 0.7798, 0.2208], 0.041400745619836664],
                [[0.2199, 0.4556, 0.6841, 0.3873, 0.9961, 0.1154, 0.9678, 0.7335, 0.1152, 0.8384], 0.05217544485450189],
                [[0.2801, 0.2895, 0.2793, 0.83, 0.4832, 0.9456, 0.139, 0.402, 0.9296, 0.1218], 0.051751059034301244],
                [[0.1197, 0.2005, 0.5692, 0.1897, 0.9014, 0.2501, 0.6541, 0.2476, 0.9808, 0.9541], 0.056797377461948295],
                [[0.17, 0.1027, 0.5433, 0.9669, 0.1256, 0.6906, 0.6202, 0.3288, 0.7465, 0.738], 0.050916190636527255],
                [[0.893, 0.4175, 0.1314, 0.1503, 0.7415, 0.5234, 0.1702, 0.7846, 0.7492, 0.1314], 0.056894089148603375],
                [[0.8962, 0.1078, 0.951, 0.8255, 0.1396, 0.129, 0.2559, 0.7041, 0.9513, 0.9961], 0.06364930131498307],
                [[0.1053, 0.4421, 0.1935, 0.1014, 0.5312, 0.7668, 0.3904, 0.958, 0.1548, 0.7625], 0.061061160204041176],
                [[0.11, 0.2594, 0.6996, 0.7597, 0.1755, 0.1002, 0.1926, 0.4621, 0.9265, 0.8734], 0.06539390880498576],
                [[0.927, 0.9604, 0.1183, 0.3875, 0.134, 0.7651, 0.8139, 0.1396, 0.5639, 0.1171], 0.06938724191363951]]
