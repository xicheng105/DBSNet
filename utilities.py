import os
import shutil
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import psutil

from torch.utils.data import Dataset
from pathlib import Path
from typing import Union, List

# %% delete_exist_file
def delete_exist_file(exist_files):
    if os.path.exists(exist_files):
        confirm = input(
            f"{exist_files} already exists. \nAre you sure you want to delete existing files? (y/n): "
        )
        if confirm.lower() != 'y':
            print("Operation aborted by user.")
            sys.exit(1)
        else:
            for file_name in os.listdir(exist_files):
                file_path = os.path.join(exist_files, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(exist_files)

# %% smart_padding
def smart_padding(kernel_size):
    if kernel_size % 2 == 0:
        return kernel_size // 2 - 1, kernel_size // 2, 0, 0
    else:
        return kernel_size // 2, kernel_size // 2, 0, 0

# %% Conv2dWithConstraint
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


# %% SmartPermute
class SmartPermute(nn.Module):
    def __init__(self):
        super(SmartPermute, self).__init__()
        self.permute_dims_5d = (0, 1, 4, 3, 2)
        self.permute_dims_4d = (0, 3, 2, 1)

    def forward(self, x):
        if x.dim() == 5:
            return x.permute(*self.permute_dims_5d)
        elif x.dim() == 4:
            return x.permute(*self.permute_dims_4d)
        else:
            raise ValueError(f'Unexpected input dimension {x.dim()} for SmartPermute')


# %% LoadData
class LoadData(Dataset):
    def __init__(
            self,
            data_dir: Union[str, Path],
            down_sampling: bool = True,
            seed: int = 42,
            subjects: List[str] = None
    ):
        """
        data_dir: 根目录。
        down_sampling: 是否进行类别均衡采样。
        seed: 随机种子。
        subjects: 指定要加载的受试者列表，如果为None则加载全部。
        """
        self.data_dir = Path(data_dir)
        self.down_sampling = down_sampling
        self.seed = seed

        if subjects is not None:
            self.files = []
            for subj in subjects:
                subj_dir = self.data_dir / subj
                self.files.extend(list(subj_dir.glob('*.npz')))
        else:
            self.files = list(self.data_dir.glob('*.npz'))

        if self.down_sampling:
            self.balanced_files = self.balance_data()
        else:
            self.balanced_files = self.files

    def __len__(self):
        return len(self.balanced_files)

    def balance_data(self):
        label_0_files = []
        label_1_files = []

        for file_path in self.files:
            data = np.load(file_path, allow_pickle=True)
            label = data['y']

            if label == 0:
                label_0_files.append(file_path)
            elif label == 1:
                label_1_files.append(file_path)

        min_samples = min(len(label_0_files), len(label_1_files))

        random.seed(self.seed)
        label_0_files = random.sample(label_0_files, min_samples)
        label_1_files = random.sample(label_1_files, min_samples)

        balanced_files = label_0_files + label_1_files
        return balanced_files

    def __getitem__(self, idx):
        file_path = self.balanced_files[idx] if self.down_sampling else self.files[idx]
        data = np.load(file_path, allow_pickle=True)

        eeg_data = data['X']
        eeg_label = data['y']
        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(eeg_label, dtype=torch.long)


# %% analyze_dataset
def analyze_dataset(dataset, dataset_name=""):
    """
    从Dataset中提取全部X和y，并打印数据形状和类别数量。

    Args:
        dataset: 传入的PyTorch Dataset对象
        dataset_name: 字符串，用于标注是train还是test集
    """
    X_list, y_list = zip(*[dataset[i] for i in range(len(dataset))])
    X = torch.stack(X_list)
    y = torch.stack(y_list)

    print(
        f"{dataset_name} Data shape: {X.shape}, Labels shape: {y.shape}\n"
        f"{dataset_name} Labels: 0 -> {(y == 0).sum().item()}, 1 -> {(y == 1).sum().item()}\n"
    )
    return X, y


# %% print_memory_usage
def print_memory_usage():
    mem = psutil.virtual_memory()
    total = round(mem.total / 1024**3, 1)
    used = round(mem.used / 1024**3, 1)
    available = round(mem.available / 1024**3, 1)
    print(f"Used memory: {used} G / Total: {total} G, Available: {available} G")

