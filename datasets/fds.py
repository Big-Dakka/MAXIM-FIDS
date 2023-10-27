###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Food Detection System Dataset
"""

import os
import shutil
import sys
from os import listdir, makedirs
from random import random

import torchvision
from torchvision import transforms

import ai8x
print(os.getcwd())
def fds_get_datasets(data, load_train=True, load_test=True):
    """
    Load FOOD 101 dataset
    """
    (data_dir, args) = data
    path = data_dir
    print("*****",path)
    dataset_path = os.path.join(path, "FDS")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        path = os.getcwd()
        print("*****",path)
        dataset_path = os.path.join(path, "data", "archive")
        is_dir = os.path.isdir(dataset_path)

        if not is_dir:
            print("******************************************")
            print("Please follow instructions below:")
            print("Dataset_path = ", dataset_path)
            print("Download the dataset in the current working directory by visiting this link"
                  "\'https://www.kaggle.com/c/dogs-vs-cats/data\'")
            print("and click the \'Download all\' button")
            print("If you do not have a Kaggle account, sign-up first.")
            print("Unzip \'archive.zip\' and you will see train.zip, test1.zip and .csv "
                  " file. Unzip the train.zip file and re-run the script")
            print("******************************************")
            sys.exit("Dataset not found..")
        else:
            # check if images set exists
            path = os.getcwd()
            dataset_path = os.path.join(path, "data", "archive", "images")
            print(dataset_path)
            is_dir = os.path.isdir(dataset_path)
            if not is_dir:
                sys.exit("Unzip \'train.zip\' file from dogs-vs-cats directory")

            # create directories
            dataset_home = os.path.join(data_dir, "FDS")
            newdir = os.path.join(dataset_home, "train", "apple_pie")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "baby_back_ribs")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "baklava")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "beef_carpaccio")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "beef_tartare")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "ceviche")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "cheesecake")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "churros")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "donuts")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "train", "french_fries")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "apple_pie")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "baby_back_ribs")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "baklava")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "beef_carpaccio")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "beef_tartare")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "ceviche")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "cheesecake")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "churros")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "donuts")
            makedirs(newdir, exist_ok=True)
            newdir = os.path.join(dataset_home, "test", "french_fries")
            makedirs(newdir, exist_ok=True)

            # define ratio of pictures to use for test set
            test_ratio = 0.2

            # copy training dataset images into subdirectories
            src_directory = os.path.join(path, "data", "archive", "images")
            for file in listdir(src_directory):
                src_folder = os.path.join(src_directory, file)
                folder = file
                for file in listdir(src_folder):
                    src = os.path.join(path, "data", "archive", "images", folder, file)
                    dst_dir = os.path.join(dataset_home, "train")
                    if random() < test_ratio:
                        dst_dir = os.path.join(dataset_home, "test")
                    dst = os.path.join(dst_dir, folder, file)
                    shutil.copyfile(src, dst)
            shutil.rmtree("archive")

    training_data_path = os.path.join(data_dir, "FDS")
    training_data_path = os.path.join(training_data_path, "train")

    test_data_path = os.path.join(data_dir, "FDS")
    test_data_path = os.path.join(test_data_path, "test")

    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=training_data_path,
                                                         transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=test_data_path,
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

datasets = [
    {
        'name': 'FDS',
        'input': (3, 64, 64),
        'output': ('apple_pie','baby_back_ribs','baklava','beef_carpaccio','beef_tartare',
            'ceviche','cheesecake','churros','donuts','french_fries'),
        'loader': fds_get_datasets,
    },
]
