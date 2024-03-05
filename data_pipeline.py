import os
import cv2
import json
import numpy as np
import torch

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import CLIPImageProcessor

class ImageDataset(Dataset):

    def __init__(self, fname_list, label_file, transform, mode=0, model_id="openai/clip-vit-base-patch32", **kwargs):
        super(ImageDataset, self).__init__(**kwargs)
        self.fname_list = fname_list
        self.transform = transform
        with open(label_file, "r") as fin:
            self.label_data = json.load(fin)

        # self.region_map = {
        #     0: [14, 38, 11, 23, 10, 7, 25, 9, 31],
        #     1: [19, 36, 16, 2, 30, 39, 17, 20],
        #     2: [41, 18, 3, 40, 29, 24, 35, 6],
        #     3: [15, 21, 8, 28, 34, 5, 13, 0, 22],
        #     4: [27, 12, 37, 1, 4, 32, 33, 26]
        # }
        # self.region = {}
        # for k,v in self.region_map.items():
        #     d = {elem:k for elem in v}
        #     self.region.update(d)

        self.name2idx = {k:idx for idx, k in enumerate(sorted(list(self.label_data.keys())))}
        if "clip" not in model_id:
            self.image_merge = self._concat_images if mode == 0 else self._stack_images
        else:
            self.processor = CLIPImageProcessor.from_pretrained(model_id)
            self.image_merge = self._clip_images

    def _clip_images(self, fnames):
        return self.processor(
            [
                torch.tensor(cv2.imread(elem)).permute(2,0,1)
                for elem in fnames
            ],
            return_tensors="pt"
        )

    def _concat_images(self, fnames):
        return torch.cat(
            [
                self.transform(torch.tensor(cv2.imread(elem)).permute(2,0,1))
                for elem in fnames
            ],
            dim=0
        )

    def _stack_images(self, fnames):
        return torch.stack(
            [
                self.transform(torch.tensor(cv2.imread(elem)).permute(2,0,1))
                for elem in fnames
            ],
            dim=1
        )

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, idx):
        curr_fname = self.fname_list[idx]
        admin1, admin2, loc_idx = curr_fname.split(os.sep)[-3:]
        geo_coords = torch.tensor(
            list(map(float, self.label_data[admin1][admin2][int(loc_idx)].split(",")))
        )
        location = os.path.abspath(self.fname_list[idx])

        fnames = [
            os.path.join(location, elem)
            for elem in sorted(os.listdir(location)) if elem.endswith("jpg")
        ]
        sample = self.image_merge(fnames)

        # class_id = self.region[self.name2idx[admin1]]
        class_id = self.name2idx[admin1]
        return sample, class_id, geo_coords

def get_geocell_centroids(train_paths, label_data, num_classes):

    centroids = np.zeros((num_classes, 2))
    cnts = defaultdict(lambda: 0)
    name2idx = {k:idx for idx, k in enumerate(sorted(list(label_data.keys())))}

    for elem in train_paths:
        county, city, idx = elem.split(os.sep)[-3:]
        lat, lon = list(map(float, label_data[county][city][int(idx)].split(",")))
        centroids[name2idx[county]] += np.array([lat, lon], dtype=np.float32)
        cnts[name2idx[county]] += 1

    for class_idx in range(num_classes):
        centroids[class_idx] /= cnts[class_idx]

    return centroids


def get_dataloaders(db_root: str, eval_splits=(0.1, 0.1), batch_size=2, random_seed=42, mode=0, model_id="openai/clip-vit-base-patch32", **kwargs):

    image_root = os.path.join(db_root, "images")
    counties = sorted(os.listdir(image_root))

    train_paths, val_paths, test_paths = [], [], []
    train_percent = 1 - sum(eval_splits)
    test_percent = eval_splits[1]

    label_file = os.path.join(db_root, "locations.json")
    with open(label_file, "r") as fin:
        label_data = json.load(fin)

    num_classes = 0
    for county in tqdm(counties):
        num_classes += 1
        cities = os.listdir(os.path.join(image_root, county))
        for city in cities:
            if city in label_data[county]:
                locations = [
                    os.path.join(image_root, county, city, elem)
                    for elem in os.listdir(os.path.join(image_root, county, city))
                ]
                cnt_locations = len(locations)
                np.random.seed(random_seed)
                np.random.shuffle(locations)
                train_paths.extend(locations[:int(cnt_locations*train_percent)])
                val_paths.extend(locations[int(cnt_locations*train_percent):-int(cnt_locations*test_percent)])
                test_paths.extend(locations[-int(cnt_locations*test_percent):])

    centroids = get_geocell_centroids(train_paths, label_data, num_classes)

    transform = v2.Compose(
        [
            v2.Lambda(lambd=lambda x: x[...,:512,:512]),
            v2.RandomCrop((128,128), pad_if_needed=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImageDataset(fname_list=train_paths, label_file=label_file, mode=mode, model_id=model_id, transform=transform)
    val_dataset = ImageDataset(fname_list=val_paths, label_file=label_file, mode=mode, model_id=model_id, transform=transform)
    test_dataset = ImageDataset(fname_list=test_paths, label_file=label_file, mode=mode, model_id=model_id, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return (train_loader, val_loader, test_loader), centroids, num_classes

if __name__=="__main__":

    (dloader, _, _), centroids, num_classes = get_dataloaders("geo_db")
    print(centroids.shape)

