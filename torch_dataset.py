import io
import os
import random

import boto3
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_world_size


def pil2bgr_numpy(image):
    """
    Convert PIL image to numpy array in BGR.

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (np.ndarray): also see `read_image`
    """
    image = image.convert("RGB")
    image = np.asarray(image)
    image = image[:, :, ::-1]
    return image


def get_bucket(s3_info):
    s3_resource = boto3.resource(
        "s3",
        endpoint_url=s3_info["endpoint_url"],
        aws_access_key_id=s3_info["aws_access_key_id"],
        aws_secret_access_key=s3_info["aws_secret_access_key"],
        region_name=s3_info["region_name"],
    )
    bucket = s3_resource.Bucket(s3_info["bucket"])
    return bucket


class S3MapDataset(data.Dataset):
    def __init__(self, s3_info):
        bucket = get_bucket(s3_info)
        self.images = [
            obj.key
            for obj in tqdm(bucket.objects.filter(Prefix=s3_info["imgroot"]))
            if obj.key.endswith(".jpg")
        ]
        self.s3_info = s3_info

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        print("path", path)
        bucket = get_bucket(self.s3_info)
        obj = bucket.Object(path)
        with io.BytesIO() as f:
            obj.download_fileobj(f)
            img = Image.open(f)
            np_img = pil2bgr_numpy(img)
        return np_img


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    print(worker_id, initial_seed + worker_id)
    seed_all_rng(initial_seed + worker_id)


class Trainer:
    def __init__(self, s3_info, is_dist=False):
        world_size = get_world_size()
        print("world size:", world_size)
        if world_size > 1:
            is_dist = True
        else:
            is_dist = False
        s3_ds = S3MapDataset(s3_info)

        sampler = DistributedSampler(s3_ds) if is_dist else None

        data_loader = data.DataLoader(
            s3_ds,
            batch_size=2,
            num_workers=2,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
            sampler=sampler,
        )
        print(len(s3_ds))

        self.dl_iter = iter(data_loader)


def main(rank):
    print(f"From rank {rank}")

    AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://play.min.io")
    AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
    AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")

    s3_info = {
        "endpoint_url": AWS_ENDPOINT_URL,
        "aws_access_key_id": AWS_ACCESS_KEY,
        "aws_secret_access_key": AWS_SECRET_ACCESS,
        "region_name": "us-east-1",
        "bucket": "coco",
        "imgroot": "coco_mini/images/",
    }

    trainer = Trainer(s3_info)
    dp = next(trainer.dl_iter)
    print("BATCH SIZE:", len(dp))


if __name__ == "__main__":
    main(0)
