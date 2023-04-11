import glob
import os
import cv2
import torch
import tqdm

import numpy as np

from segment_anything import sam_model_registry, SamPredictor
from monai.transforms import LoadImaged, ScaleIntensityRanged, Compose, Identity
from monai.data import DataLoader, Dataset, decollate_batch
from monai.utils import first
from monai.metrics import DiceMetric
from utils import utils

data_dir = '/mnt/B-SSD/maltamed/datasets/unprocessed/ISIC/'
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files = data_dicts[1:2]

transform = None
train_transforms = Compose(
    LoadImaged(keys=["image", "label"]),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0, b_max=255, clip=True) if transform =='liver' else Identity(keys=["image", "label"]),
)

check_ds = Dataset(data=train_files, transform=train_transforms)
loader = DataLoader(check_ds, batch_size=1)

sam_checkpoint = "/mnt/B-SSD/maltamed/SAM-zero-shot/segment-anything-main/checkpoints/sam_vit_h.pth"
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

dice_metric_0 = DiceMetric(include_background=True, reduction="mean")
dice_metric_1 = DiceMetric(include_background=True, reduction="mean")
dice_metric_2 = DiceMetric(include_background=True, reduction="mean")
for batch in tqdm.tqdm(loader):
    image, mask = batch["image"][0], batch["label"][0]
    mask = mask.numpy()
    non_zero_slices = np.any(mask, axis=(0, 1))
    mask = mask[:,:,non_zero_slices]
    image = image[:,:,non_zero_slices]

    mask = np.where(mask > 0, 1, 0)
    image = np.array(image)
    image = np.clip(image, 0, 255)

    for i in range(image.shape[-1]):
        mask_ = mask[:,:,i]
        image_ = image[:,:,i]
        coordinate = utils.random_coordinate(mask)
        image_ = np.array(image_).astype(np.uint8)
        mask_ = np.array(mask_).astype(np.uint8)

        mask_ = cv2.cvtColor(mask_, cv2.COLOR_GRAY2RGB)
        image_ = cv2.cvtColor(image_, cv2.COLOR_GRAY2RGB)
        
        predictor.set_image(image_)

        input_point = np.array([[coordinate[1], coordinate[0]]])
        input_label = np.array([1])
        masks_pred, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        masks_pred_0_1 = masks_pred[0,:,:] #+ masks_pred[1,:,:] #+ masks_pred[2,:,:]
        dice_metric_0(y_pred=torch.Tensor(masks_pred[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask_[:,:,0]*1).unsqueeze(0))
        dice_metric_1(y_pred=torch.Tensor((masks_pred[0,:,:]+masks_pred[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask_[:,:,0]*1).unsqueeze(0))
        dice_metric_2(y_pred=torch.Tensor((masks_pred[0,:,:]+masks_pred[1,:,:]+masks_pred[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask_[:,:,0]*1).unsqueeze(0))

dice_0 = dice_metric_0.aggregate().item()
dice_1 = dice_metric_1.aggregate().item()
dice_2 = dice_metric_2.aggregate().item()

dice_metric_0.reset()
dice_metric_1.reset()
dice_metric_2.reset()