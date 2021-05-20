
import h5py
import torch
import random

# random label sampling
class H5Dataset_random(torch.utils.data.Dataset):
    def __init__(self, file_path, seed=None):
        self.h5_file = h5py.File(file_path, 'r')

        self.case_list = sorted(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.label_list = sorted(set([k.split('_')[3] for k in self.h5_file.keys() if len(k.split('_')) > 3]))

        self.num_cases = len(self.case_list)
        self.num_labels = len(self.label_list)
        self.num_frames = [int(len([k for k in self.h5_file.keys() if k.split('_')[1] == idx_case])/(1+self.num_labels)) for idx_case in self.case_list]

        self.seed = seed

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx_case):
        # for consistent indexing
        if self.seed is not None:
            random.seed(self.seed)

        # sampling random frame
        idx_frame = random.randint(0, self.num_frames[idx_case] - 1)
        frame = torch.unsqueeze(
            torch.tensor(self.h5_file.get('/frame_{0:04d}_{1:03d}'.format(idx_case, idx_frame))[()].astype('float32')),
            dim=0)

        # sampling random label
        idx_label = random.randint(0, self.num_labels - 1)
        label_1 = torch.unsqueeze(torch.tensor(self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(idx_case, idx_frame, idx_label))[()].astype('int64')), dim=0)

        # sampling consensus label classification
        label_2 =  [torch.unsqueeze(torch.tensor(
                self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(idx_case, idx_frame, idx_label))[()].astype('int64')), dim=0)
             for idx_label in range(self.num_labels)]
        count = 0
        for i in label_2:
            if torch.count_nonzero(i) != 0:
                count += 1
        if count >= 2:
            label_2 = torch.tensor([1, 0]).float()
        else:
            label_2 = torch.tensor([0, 1]).float()

        return (frame, label_1, label_2)

    # helper
    def dim(self):
        x = torch.tensor(self.h5_file['/frame_0000_000'][()])
        return x.shape


# consensus label sampling (majority voting at the pixel level)
class H5Dataset_consensus(torch.utils.data.Dataset):
    def __init__(self, file_path, seed=None):
        self.h5_file = h5py.File(file_path, 'r')

        self.case_list = sorted(set([k.split('_')[1] for k in self.h5_file.keys()]))
        self.label_list = sorted(set([k.split('_')[3] for k in self.h5_file.keys() if len(k.split('_')) > 3]))

        self.num_cases = len(self.case_list)
        self.num_labels = len(self.label_list)
        self.num_frames = [int(len([k for k in self.h5_file.keys() if k.split('_')[1] == idx_case])/(1+self.num_labels)) for idx_case in self.case_list]

        self.seed = seed

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx_case):
        # for consistent indexing
        if self.seed is not None:
            random.seed(self.seed)

        # sampling random frame
        idx_frame = random.randint(0, self.num_frames[idx_case] - 1)
        frame = torch.unsqueeze(torch.tensor(self.h5_file.get('/frame_{0:04d}_{1:03d}'.format(idx_case, idx_frame))[()].astype('float32')), dim=0)

        # sampling consensus label at pixel level
        label_1 = (torch.stack(
            [torch.unsqueeze(torch.tensor(
                self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(idx_case, idx_frame, idx_label))[()].astype('int64')), dim=0)
             for idx_label in range(self.num_labels)],
            dim=0
        ).sum(dim=0) >= self.num_labels / 2).long()

        # sampling consensus label classification
        label_2 =  [torch.unsqueeze(torch.tensor(
                self.h5_file.get('/label_{0:04d}_{1:03d}_{2:02d}'.format(idx_case, idx_frame, idx_label))[()].astype('int64')), dim=0)
             for idx_label in range(self.num_labels)]
        count = 0
        for i in label_2:
            if torch.count_nonzero(i) != 0:
                count += 1
        if count >= 2:
            label_2 = torch.tensor([1, 0]).float()
        else:
            label_2 = torch.tensor([0, 1]).float()

        return (frame, label_1, label_2)

    # helper
    def dim(self):
        x = torch.tensor(self.h5_file['/frame_0000_000'][()])
        return x.shape
