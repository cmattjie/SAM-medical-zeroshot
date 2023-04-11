import glob
import os
import cv2
import torch
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt 
import numpy as np

from segment_anything import sam_model_registry, SamPredictor
from monai.transforms import LoadImaged, ScaleIntensityRanged, Compose, Identity
from monai.data import DataLoader, Dataset, ThreadDataLoader
from monai.metrics import DiceMetric
from utils import utils
import warnings
import json
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

def get_args():
    parser = argparse.ArgumentParser(description='parameters for evaluating SAM')
    
    parser.add_argument('--model',              default='default', type=str, help='Model name.')
    parser.add_argument('--dataset',            default='ISIC', type=str, help='dataset')
    parser.add_argument('--gpu',                default=0, type=str, help='GPU Number.')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    #parser.add_argument('--mask_mode',          default='rnd', type=str, help='Method for sampling points.')
    #parser.add_argument('--n_splits',           default=3, type=int, help='Number of splits to get points of.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    return args 

args = get_args()
dicts = json.load(open('./utils/dicts.json', 'r'))
board = SummaryWriter(log_dir='./runs/'+args.name)

assert args.dataset in dicts['dataset_processed_path'].keys(), "Dataset not found"
assert args.mask_mode in ['rnd', 'split', 'central', 'box'], "Method not found, choose between rnd, 3split, central, box"
assert args.model in dicts['sam_checkpoint'].keys(), "Model not found"

data_dir = dicts['dataset_processed_path'][args.dataset]
train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.*")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.*")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files = data_dicts

# transform = None
# train_transforms = Compose(
#     LoadImaged(keys=["image", "label"]),
#     ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0, b_max=255, clip=True) if transform =='liver' else Identity(),
# )

check_ds = Dataset(data=train_files)# transform=train_transforms)
loader = DataLoader(check_ds, batch_size=1)#, num_workers=4, shuffle=False)

sam_checkpoint = dicts['sam_checkpoint'][args.model]
device = "cuda"
model_type = args.model

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

dice_rc_0, dice_rc_1, dice_rc_2 = DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean")
dice_rs3_0, dice_rs3_1, dice_rs3_2 = DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean")
dice_rs5_0, dice_rs5_1, dice_rs5_2 = DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean")
dice_cp_0, dice_cp_1, dice_cp_2 = DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean")
dice_bb_0, dice_bb_1, dice_bb_2 = DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean"), DiceMetric(include_background=True, reduction="mean")

#loop with tqdm
n_images = len(loader)

#get 10 random values from n_images
random_values = np.random.randint(0, n_images, 10)
print(f"Random values: {random_values}")
 
for idx, batch in enumerate(tqdm(loader)):
    image_loc, mask_loc = batch["image"][0], batch["label"][0]

    #reading images and mask
    image = cv2.imread(image_loc, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_loc, cv2.IMREAD_GRAYSCALE)
    
    #FILL HOLES on mask
    contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask,[cnt],0,255,-1)
    
    #get points ans boxes
    input_rc, input_label_rc = utils.random_coordinate(mask)
    input_rs3, input_label_rs3 = utils.random_splits(mask, n_splits=3)
    input_rs5, input_label_rs5 = utils.random_splits(mask, n_splits=5)
    input_cp, input_label_cp = utils.central_point(mask)
    input_bb = utils.boundbox(mask)
    
    predictor.set_image(image)
    
    #predict masks from points and boxes
    masks_rc, scores_rc, logits_rc = predictor.predict(point_coords=input_rc, point_labels=input_label_rc,multimask_output=True)
    masks_rs3, scores_rs3, logits_rs3 = predictor.predict(point_coords=input_rs3, point_labels=input_label_rs3,multimask_output=True)
    masks_rs5, scores_rs5, logits_rs5 = predictor.predict(point_coords=input_rs5, point_labels=input_label_rs5,multimask_output=True)
    masks_cp, scores_cp, logits_cp = predictor.predict(point_coords=input_cp, point_labels=input_label_cp,multimask_output=True)
    masks_bb, scores_bb, logits_bb = predictor.predict(point_coords=None, point_labels=None, box=input_bb, multimask_output=True)

    if idx in random_values:
        #save images
        #invert channels for tensorboard
        image = image[:,:,::-1]
        board.add_image(f'images_{idx}/original', image, idx, dataformats='HWC')
        board.add_image(f'images_{idx}/mask', mask, idx, dataformats='HW')
        # plot_2d_or_3d_image(image, idx, board, index=0, tag='images/original')
        # plot_2d_or_3d_image(mask, idx, board, index=0, tag='images/mask')
        print(mask_loc)
        print('MASKS_R SHAPE:',masks_rc.shape)
        
        for i in range(3):
            #TODO ADD POINTS AND BOUNDING BOXES
            board.add_image(f'images_{idx}/mask_rc_{i}', masks_rc[i,:,:], idx, dataformats='HW')
            board.add_image(f'images_{idx}/mask_rs3_{i}', masks_rs3[i,:,:], idx, dataformats='HW')
            board.add_image(f'images_{idx}/mask_rs5_{i}', masks_rs5[i,:,:], idx, dataformats='HW')
            board.add_image(f'images_{idx}/mask_cp_{i}', masks_cp[i,:,:], idx, dataformats='HW')
            board.add_image(f'images_{idx}/mask_bb_{i}', masks_bb[i,:,:], idx, dataformats='HW')
            # plot_2d_or_3d_image(masks_rc[i,:,:], idx, board, index=0, tag=f'images/mask_rc_{i}')
            # plot_2d_or_3d_image(masks_rs3[i,:,:], idx, board, index=0, tag=f'images/mask_rs3_{i}')
            # plot_2d_or_3d_image(masks_rs5[i,:,:], idx, board, index=0, tag=f'images/mask_rs5_{i}')
            # plot_2d_or_3d_image(masks_cp[i,:,:], idx, board, index=0, tag=f'images/mask_cp_{i}')
            # plot_2d_or_3d_image(masks_bb[i,:,:], idx, board, index=0, tag=f'images/mask_bb_{i}')
    
    #calculate metrics
    dice_rc_0(y_pred=torch.Tensor(masks_rc[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rc_1(y_pred=torch.Tensor((masks_rc[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rc_2(y_pred=torch.Tensor((masks_rc[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs3_0(y_pred=torch.Tensor(masks_rs3[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs3_1(y_pred=torch.Tensor((masks_rs3[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs3_2(y_pred=torch.Tensor((masks_rs3[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs5_0(y_pred=torch.Tensor(masks_rs5[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs5_1(y_pred=torch.Tensor((masks_rs5[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_rs5_2(y_pred=torch.Tensor((masks_rs5[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_cp_0(y_pred=torch.Tensor(masks_cp[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_cp_1(y_pred=torch.Tensor((masks_cp[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_cp_2(y_pred=torch.Tensor((masks_cp[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_bb_0(y_pred=torch.Tensor(masks_bb[0,:,:]*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_bb_1(y_pred=torch.Tensor((masks_bb[1,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    dice_bb_2(y_pred=torch.Tensor((masks_bb[2,:,:])*1).unsqueeze(0), y=torch.Tensor(mask[:,:]*1).unsqueeze(0))
    
dice_rc_0 = dice_rc_0.aggregate().item()
dice_rc_1 = dice_rc_1.aggregate().item()
dice_rc_2 = dice_rc_2.aggregate().item()
dice_rs3_0 = dice_rs3_0.aggregate().item()
dice_rs3_1 = dice_rs3_1.aggregate().item()
dice_rs3_2 = dice_rs3_2.aggregate().item()
dice_rs5_0 = dice_rs5_0.aggregate().item()
dice_rs5_1 = dice_rs5_1.aggregate().item()
dice_rs5_2 = dice_rs5_2.aggregate().item()
dice_cp_0 = dice_cp_0.aggregate().item()
dice_cp_1 = dice_cp_1.aggregate().item()
dice_cp_2 = dice_cp_2.aggregate().item()
dice_bb_0 = dice_bb_0.aggregate().item()
dice_bb_1 = dice_bb_1.aggregate().item()
dice_bb_2 = dice_bb_2.aggregate().item()

board.add_scalar('Dice Random Coord 0', dice_rc_0, 0)
board.add_scalar('Dice Random Coord 1', dice_rc_1, 1)
board.add_scalar('Dice Random Coord 2', dice_rc_2, 2) 
board.add_scalar('Dice 3 Random Splits 0', dice_rs3_0, 0)
board.add_scalar('Dice 3 Random Splits 1', dice_rs3_1, 1)
board.add_scalar('Dice 3 Random Splits 2', dice_rs3_2, 2)
board.add_scalar('Dice 5 Random Splits 0', dice_rs5_0, 0)
board.add_scalar('Dice 5 Random Splits 1', dice_rs5_1, 1)
board.add_scalar('Dice 5 Random Splits 2', dice_rs5_2, 2)
board.add_scalar('Dice Central Point 0', dice_cp_0, 0)
board.add_scalar('Dice Central Point 1', dice_cp_1, 1)
board.add_scalar('Dice Central Point 2', dice_cp_2, 2)
board.add_scalar('Dice Bounding Box 0', dice_bb_0, 0)
board.add_scalar('Dice Bounding Box 1', dice_bb_1, 1)
board.add_scalar('Dice Bounding Box 2', dice_bb_2, 2)

 