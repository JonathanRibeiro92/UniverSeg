from tqdm import tqdm
import h5py
import nibabel as nib
import sys
import os
from os import listdir
from os.path import isdir, join

import numpy as np
from glob import glob
from torch.utils.data import Dataset
from dataclasses import dataclass
import subprocess
import pathlib
from typing import Literal, Tuple
import PIL
import torch


output_size =[128, 128, 80]


def require_download_msd():
    dest_folder = pathlib.Path("./tmp/universeg_msd/")

    if not dest_folder.exists():
        tar_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
        subprocess.run(
            ["curl", tar_url, "--create-dirs", "-o",
                str(dest_folder/'Task01_BrainTumour.tar'),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["tar", 'xf', str(
                dest_folder/'Task01_BrainTumour.tar'), '-C', str(dest_folder)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder

def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = nib.load(path).get_fdata().astype(np.int8).squeeze()
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()

def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("imagesTr/*.nii.gz")):
        img = process_img(file, size=size)
        seg_file = pathlib.Path(str(file).replace("imagesTr", "labelsTr"))
        seg = process_seg(seg_file, size=size)
        data.append((img, seg))
    return data


@dataclass
class MSDDataset(Dataset):
    split: Literal["support", "test"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        path = require_download_msd()
        currentDir = os.path.abspath(os.getcwd())
        path = currentDir / path
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.label is not None:
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]
        return img, seg


def covert_h5(root):
    listt = glob( os.path.join(root, 'imagesTr/*.nii.gz') )
    do_localization = False
    
    for item in tqdm(listt):
        image_obj = nib.load(item)
        image = image_obj.get_fdata()
        label_obj = nib.load(item.replace('imagesTr', 'labelsTr'))
        label = label_obj.get_fdata()
        label = label.astype(np.uint8)
        # label = (label >= 1).astype(np.uint8)
        w, h, d = label.shape
        image_shape = image.shape

        if do_localization:
            tempL = np.nonzero(label)
            # Find the boundary of non-zero labels
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])

            # px, py, pz ensure the output image is at least of output_size
            px = max(output_size[0] - (maxx - minx), 0) // 2
            py = max(output_size[1] - (maxy - miny), 0) // 2
            pz = max(output_size[2] - (maxz - minz), 0) // 2
            # randint(10, 20) lets randomly-sized zero margins included in the output image
            minx = max(minx - np.random.randint(10, 20) - px, 0)
            maxx = min(maxx + np.random.randint(10, 20) + px, w)
            miny = max(miny - np.random.randint(10, 20) - py, 0)
            maxy = min(maxy + np.random.randint(10, 20) + py, h)
            minz = max(minz - np.random.randint(5, 10) - pz, 0)
            maxz = min(maxz + np.random.randint(5, 10) + pz, d)
        else:
            tempL = np.nonzero(image)
            # Find the boundary of non-zero labels
            minx, maxx = np.min(tempL[0]), np.max(tempL[0])
            miny, maxy = np.min(tempL[1]), np.max(tempL[1])
            minz, maxz = np.min(tempL[2]), np.max(tempL[2])
                        
        image = image[minx:maxx, miny:maxy, minz:maxz]
        image = image.astype(np.float32)
        if len(image.shape) == 4:
            MOD = image.shape[3]
            for m in range(MOD):
                image[:, :, :, m] = (image[:, :, :, m] - np.mean(image[:, :, :, m])) / np.std(image[:, :, :, m])
        else:
            image = (image - np.mean(image)) / np.std(image)
            
        label = label[minx:maxx, miny:maxy, minz:maxz]
        print("%s: %s => %s, %s" %(item, image_shape, image.shape, label.shape))

        f = h5py.File(item.replace('.nii.gz', '.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


if __name__ == '__main__':

    covert_h5(sys.argv[1])
    