import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root_path, label_path):
        self.label_df = pd.read_excel(label_path, sheet_name='Sheet1')
        self.views = ['dhw', 'hdw', 'whd']
        # self.views = ['dhw']
        self.data_path_with_label = []
        for i in range(len(self.label_df)):
            label = self.label_df['GOLDCLA'][i] - 1
            subject = self.label_df['subject'][i]
            image_path = os.path.join(data_root_path, subject + '_dhw_128.npy')
            if os.path.exists(image_path):
                for view in self.views:
                    self.data_path_with_label.append({'image_path': image_path, 'label': label, 'subject': subject, 'view': view})

        random.shuffle(self.data_path_with_label)

    def __len__(self):
        return len(self.data_path_with_label)

    def __getitem__(self, idx):
        path = self.data_path_with_label[idx]['image_path']
        label = self.data_path_with_label[idx]['label']
        subject = self.data_path_with_label[idx]['subject']
        if path[-4:] == '.npy':
            image_array = np.load(path)  # (1,D,H,W)
            if self.data_path_with_label[idx]['view'] == 'hdw':
                image_array = image_array.swapaxes(1, 2)
            elif self.data_path_with_label[idx]['view'] == 'whd':
                image_array = image_array.swapaxes(1, 3)

        return torch.from_numpy(image_array).float(), label, subject
