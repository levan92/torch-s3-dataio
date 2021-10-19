import io
import os

import boto3
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


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


AWS_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "https://play.min.io")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS = os.environ.get("AWS_SECRET_ACCESS")


class S3MapDataset(data.Dataset):
    def __init__(self, bucket="coco", imgroot="coco_mini/images/"):
        assert AWS_ACCESS_KEY is not None
        print(AWS_ENDPOINT_URL)
        print(AWS_ACCESS_KEY)
        print(AWS_SECRET_ACCESS)
        s3_resource = boto3.resource(
            "s3",
            endpoint_url=AWS_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS,
            region_name="us-east-1",
        )
        self.bucket = s3_resource.Bucket(bucket)
        self.images = [
            obj.key
            for obj in tqdm(self.bucket.objects.filter(Prefix=imgroot))
            if obj.key.endswith(".jpg")
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        print("path", path)
        obj = self.bucket.Object(path)
        with io.BytesIO() as f:
            obj.download_fileobj(f)
            img = Image.open(f)
            print(img)
            np_img = pil2bgr_numpy(img)
        print(np_img.shape)
        return np_img


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    print(worker_id, initial_seed)


s3_ds = S3MapDataset()

data_loader = data.DataLoader(
    s3_ds,
    batch_size=2,
    num_workers=2,
    collate_fn=trivial_batch_collator,
    worker_init_fn=worker_init_reset_seed,
)

dl_iter = iter(data_loader)
