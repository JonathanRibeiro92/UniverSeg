from tqdm import tqdm
import h5py
import nrrd
import nibabel as nib
import sys
import os
from os import listdir
from os.path import isdir, join

import numpy as np
from glob import glob
from torch.utils.data import Dataset
from dataclasses import dataclass


output_size =[128, 128, 80]


def localize(image, mask, min_output_size):
    if type(min_output_size) == int:
        H = W = D = min_output_size
    else:
        H, W, D = min_output_size

    tempL = np.nonzero(mask)
    # Find the boundary of non-zero mask
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])

    # px, py, pz ensure the output image is at least of min_output_size
    px = max(min_output_size[0] - (maxx - minx), 0) // 2
    py = max(min_output_size[1] - (maxy - miny), 0) // 2
    pz = max(min_output_size[2] - (maxz - minz), 0) // 2
    # randint(10, 20) lets randomly-sized zero margins included in the output image
    minx = max(minx - np.random.randint(10, 20) - px, 0)
    maxx = min(maxx + np.random.randint(10, 20) + px, H)
    miny = max(miny - np.random.randint(10, 20) - py, 0)
    maxy = min(maxy + np.random.randint(10, 20) + py, W)
    minz = max(minz - np.random.randint(5, 10) - pz, 0)
    maxz = min(maxz + np.random.randint(5, 10) + pz, D)

    if len(image.shape) == 4:
        image = image[:, minx:maxx, miny:maxy, minz:maxz]
    else:
        image = image[minx:maxx, miny:maxy, minz:maxz]

    mask = mask[minx:maxx, miny:maxy, minz:maxz]
    return image, mask

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

@dataclass
class BratsSet(Dataset):
    """ Annual Brats challenges dataset """

    # binarize: whether to binarize mask (do whole-tumor segmentation)
    # modality: if the image has multiple modalities,
    # choose which modality to output (-1 to output all).
    # If mode == 'train' and train_loc_prob > 0, then min_output_size is necessary.
    def __init__(self, base_dir, split, mode, sample_num=None,
                 ds_weight=1.,
                 xyz_permute=None, transform=None,
                 chosen_modality=-1, binarize=False,
                 train_loc_prob=0, min_output_size=None):
        super(BratsSet, self).__init__()
        self._base_dir = base_dir
        self.split = split
        self.mode = mode
        self.xyz_permute = xyz_permute
        self.ds_weight = ds_weight

        self.transform = transform
        self.chosen_modality = chosen_modality
        self.binarize = binarize
        self.train_loc_prob = train_loc_prob
        self.min_output_size = min_output_size

        trainlist_filepath = self._base_dir + '/train.list'
        testlist_filepath = self._base_dir + '/test.list'
        alllist_filepath = self._base_dir + '/all.list'

        if not os.path.isfile(alllist_filepath):
            self.create_file_list(0.85)

        with open(trainlist_filepath, 'r') as f:
            self.train_image_list = f.readlines()
        with open(testlist_filepath, 'r') as f:
            self.test_image_list = f.readlines()
        with open(alllist_filepath, 'r') as f:
            self.all_image_list = f.readlines()

        if self.split == 'train':
            self.image_list = self.train_image_list
        elif self.split == 'test':
            self.image_list = self.test_image_list
        elif self.split == 'all':
            self.image_list = self.all_image_list

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if sample_num is not None:
            self.image_list = self.image_list[:sample_num]

        self.num_modalities = 0
        # Fetch image 0 to get num_modalities.
        sample0 = self.__getitem__(0, do_transform=False)
        if len(sample0['image'].shape) == 4:
            self.num_modalities = sample0['image'].shape[0]

        print("'{}' {} samples, num_modalities: {}, chosen: {}".format(self.split,
                                                                       len(self.image_list), self.num_modalities,
                                                                       self.chosen_modality))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, do_transform=True):
        image_name = self.image_list[idx]
        image_path = os.path.join(self._base_dir, image_name)
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        mask = h5f['label'][:]
        if self.num_modalities > 0 and self.chosen_modality != -1:
            image = image[self.chosen_modality, :, :, :]
        if self.binarize:
            mask = (mask >= 1).astype(np.uint8)
        else:
            # Map 4 to 3, and keep 0,1,2 unchanged.
            mask -= (mask == 4)

        if self.mode == 'train' and self.train_loc_prob > 0 \
                and np.random.random() < self.train_loc_prob:
            image, mask = localize(image, mask, self.min_output_size)

        # xyz_permute by default is None.
        if do_transform and self.xyz_permute is not None:
            image = image.transpose(self.xyz_permute)
            mask = mask.transpose(self.xyz_permute)

        sample = {'image': image, 'mask': mask}
        if do_transform and self.transform:
            sample = self.transform(sample)

        sample['image_path'] = image_name
        sample['weight'] = self.ds_weight
        return sample

    def create_file_list(self, train_test_split):
        img_dirs = [d for d in listdir(self._base_dir) if isdir(join(self._base_dir, d))]

        # Randomize the file list, then split. Not to use the official testing set
        # since we don't have ground truth masks for this.
        num_files = len(img_dirs)
        idxList = np.arange(num_files)  # List of file indices
        self.imgFiles = {}
        for idx in idxList:
            self.imgFiles[idx] = join(img_dirs[idx], img_dirs[idx] + ".h5")

        with open(join(self._base_dir, 'all.list'), "w") as allFile:
            for img_idx in idxList:
                allFile.write("%s\n" % self.imgFiles[img_idx])
        allFile.close()

        idxList = np.random.permutation(idxList)  # Randomize list
        train_len = int(np.floor(num_files * train_test_split))  # Number of training files
        train_indices = idxList[0:train_len]  # List of training indices
        test_indices = idxList[train_len:]  # List of testing indices

        with open(join(self._base_dir, 'train.list'), "w") as trainFile:
            for img_idx in sorted(train_indices):
                trainFile.write("%s\n" % self.imgFiles[img_idx])
        trainFile.close()

        with open(join(self._base_dir, 'test.list'), "w") as testFile:
            for img_idx in sorted(test_indices):
                testFile.write("%s\n" % self.imgFiles[img_idx])
        testFile.close()

        print("%d files are split to %d training, %d test" % (num_files, train_len, len(test_indices)))

if __name__ == '__main__':
    covert_h5(sys.argv[1])
    