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
    parser.add_argument('--gpu',                default=2, type=str, help='GPU Number.')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    #parser.add_argument('--mask_mode',          default='rnd', type=str, help='Method for sampling points.')
    #parser.add_argument('--n_splits',           default=3, type=int, help='Number of splits to get points of.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    return args 

args = get_args()
dicts = json.load(open('./utils/dicts.json', 'r'))
board = SummaryWriter(log_dir='./runs/'+args.name)

assert args.dataset in dicts['dataset_processed_path'].keys(), "Dataset not found"
#assert args.mask_mode in ['rnd', 'split', 'central', 'box'], "Method not found, choose between rnd, 3split, central, box"
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
loader = DataLoader(check_ds, batch_size=1, shuffle=True)#, num_workers=4, shuffle=False)

sam_checkpoint = dicts['sam_checkpoint'][args.model]
device = torch.device(f'cuda:{args.gpu}')
print(device)
model_type = args.model

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

dice_rc_0 = DiceMetric(include_background=True, reduction="mean")
dice_rc_1 = DiceMetric(include_background=True, reduction="mean")
dice_rc_2 = DiceMetric(include_background=True, reduction="mean")
dice_rs3_0 = DiceMetric(include_background=True, reduction="mean")
dice_rs3_1 = DiceMetric(include_background=True, reduction="mean")
dice_rs3_2 = DiceMetric(include_background=True, reduction="mean")
dice_rs5_0 = DiceMetric(include_background=True, reduction="mean")
dice_rs5_1 = DiceMetric(include_background=True, reduction="mean")
dice_rs5_2 = DiceMetric(include_background=True, reduction="mean")
dice_cp_0 = DiceMetric(include_background=True, reduction="mean")
dice_cp_1 = DiceMetric(include_background=True, reduction="mean")
dice_cp_2 = DiceMetric(include_background=True, reduction="mean")
dice_bb_0 = DiceMetric(include_background=True, reduction="mean")
dice_bb_1 = DiceMetric(include_background=True, reduction="mean")
dice_bb_2 = DiceMetric(include_background=True, reduction="mean")
dice_bbs_0 = DiceMetric(include_background=True, reduction="mean")
dice_bbs_1 = DiceMetric(include_background=True, reduction="mean")
dice_bbs_2 = DiceMetric(include_background=True, reduction="mean")

#TODO ARRUMAR ERODE E ARRUMAR VARIAÇÃO BOUNDING BOX
#loop with tqdm
n_images = len(loader)

#get 10 random values from n_images
random_values = [1,2,3,4,5]#np.random.randint(0, n_images, 10)
#print(f"Random values: {random_values}")

for idx, batch in enumerate(tqdm(loader)):
    # if idx < 3680:
    #    continue
    
    image_loc, mask_loc = batch["image"][0], batch["label"][0]

    #reading images and mask
    image = cv2.imread(image_loc, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_loc, cv2.IMREAD_GRAYSCALE)
    
    #FILL HOLES on mask
    if args.dataset == 'ISIC':
        contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask,[cnt],0,255,-1)
    
    masks_split = utils.split_mask(mask, args.dataset)
    
    if args.dataset == "CXRkaggle" and len(masks_split) !=2:
        print(f"only 1 lung mask found in image: {image_loc}")
        
    #create lists and set image
    masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs = [], [], [], [], [], []
    
    predictor.set_image(image)
    
    for mask_s in masks_split:
        #get points and boxes
        try:
            input_rc, input_label_rc = utils.random_coordinate(mask_s)
            input_rs3, input_label_rs3 = utils.random_splits(mask_s, n_splits=3)
            input_rs5, input_label_rs5 = utils.random_splits(mask_s, n_splits=5)
            input_cp, input_label_cp = utils.central_point(mask_s)
            input_bb, input_bb_similar = utils.boundbox_similar(mask_s, mask_s.shape)
     
            #predict masks from points and boxes
            masks_rc_temp, scores_rc, logits_rc = predictor.predict(point_coords=input_rc, point_labels=input_label_rc,multimask_output=True)
            masks_rs3_temp, scores_rs3, logits_rs3 = predictor.predict(point_coords=input_rs3, point_labels=input_label_rs3,multimask_output=True)
            masks_rs5_temp, scores_rs5, logits_rs5 = predictor.predict(point_coords=input_rs5, point_labels=input_label_rs5,multimask_output=True)
            masks_cp_temp, scores_cp, logits_cp = predictor.predict(point_coords=input_cp, point_labels=input_label_cp,multimask_output=True)
            masks_bb_temp, scores_bb, logits_bb = predictor.predict(point_coords=None, point_labels=None, box=input_bb, multimask_output=True)
            masks_bbs_temp, scores_bbs, logits_bbs = predictor.predict(point_coords=None, point_labels=None, box=input_bb_similar, multimask_output=True)
            
            #append masks
            masks_rc.append(masks_rc_temp)
            masks_rs3.append(masks_rs3_temp)
            masks_rs5.append(masks_rs5_temp)
            masks_cp.append(masks_cp_temp)
            masks_bb.append(masks_bb_temp)
            masks_bbs.append(masks_bbs_temp)
            
        except:
            #TODO SAVE COMPRESSED VERSION OF IMAGE
            print(f"Error in image {idx}")
            print(mask_loc)
            board.add_image(f'error_{mask_loc.split("/")[-1]}/original', image, idx, dataformats='HWC')
            board.add_image(f'error_{mask_loc.split("/")[-1]}/mask', mask, idx, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_rc', np.expand_dims(masks_rc*1, axis=-1), 0, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_rs3', np.expand_dims(masks_rs3*1, axis=-1), 0, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_rs5', np.expand_dims(masks_rs5*1, axis=-1), 0, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_cp', np.expand_dims(masks_cp*1, axis=-1), 0, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_bb', np.expand_dims(masks_bb*1, axis=-1), 0, dataformats='HW')
            board.add_images(f'error_{mask_loc.split("/")[-1]}/mask_bbs', np.expand_dims(masks_bbs*1, axis=-1), 0, dataformats='HW')
            continue
    #lista (5 elementos) de listas (2 elementos) de listas de 3 predições de máscara
    mask_list = [masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs]
    masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs = utils.merge_masks(mask_list)
    
    if idx in random_values:
        #save images
        #invert channels for tensorboard
        image = image[:,:,::-1]
        board.add_image(f'images_{mask_loc.split("/")[-1]}/original', image, idx, dataformats='HWC')
        board.add_image(f'images_{mask_loc.split("/")[-1]}/mask', mask, idx, dataformats='HW')
        # plot_2d_or_3d_image(image, idx, board, index=0, tag='images/original')
        # plot_2d_or_3d_image(mask, idx, board, index=0, tag='images/mask')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_rc', np.expand_dims(masks_rc*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_rs3', np.expand_dims(masks_rs3*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_rs5', np.expand_dims(masks_rs5*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_cp', np.expand_dims(masks_cp*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_bb', np.expand_dims(masks_bb*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_bbs', np.expand_dims(masks_bbs*1, axis=-1), 0, dataformats='NHWC')
    
    mask = (mask>0)*1

    dice_rc_0(y_pred=torch.Tensor(masks_rc[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rc_1(y_pred=torch.Tensor((masks_rc[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rc_2(y_pred=torch.Tensor((masks_rc[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs3_0(y_pred=torch.Tensor(masks_rs3[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs3_1(y_pred=torch.Tensor((masks_rs3[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs3_2(y_pred=torch.Tensor((masks_rs3[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs5_0(y_pred=torch.Tensor(masks_rs5[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs5_1(y_pred=torch.Tensor((masks_rs5[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_rs5_2(y_pred=torch.Tensor((masks_rs5[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_cp_0(y_pred=torch.Tensor(masks_cp[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_cp_1(y_pred=torch.Tensor((masks_cp[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_cp_2(y_pred=torch.Tensor((masks_cp[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bb_0(y_pred=torch.Tensor(masks_bb[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bb_1(y_pred=torch.Tensor((masks_bb[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bb_2(y_pred=torch.Tensor((masks_bb[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_0(y_pred=torch.Tensor(masks_bbs[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_1(y_pred=torch.Tensor((masks_bbs[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_2(y_pred=torch.Tensor((masks_bbs[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    
dice_rc_0 = dice_rc_0.aggregate()
dice_rc_1 = dice_rc_1.aggregate()
dice_rc_2 = dice_rc_2.aggregate()
dice_rs3_0 = dice_rs3_0.aggregate()
dice_rs3_1 = dice_rs3_1.aggregate()
dice_rs3_2 = dice_rs3_2.aggregate()
dice_rs5_0 = dice_rs5_0.aggregate()
dice_rs5_1 = dice_rs5_1.aggregate()
dice_rs5_2 = dice_rs5_2.aggregate()
dice_cp_0 = dice_cp_0.aggregate()
dice_cp_1 = dice_cp_1.aggregate()
dice_cp_2 = dice_cp_2.aggregate()
dice_bb_0 = dice_bb_0.aggregate()
dice_bb_1 = dice_bb_1.aggregate()
dice_bb_2 = dice_bb_2.aggregate()
dice_bbs_0 = dice_bbs_0.aggregate()
dice_bbs_1 = dice_bbs_1.aggregate()
dice_bbs_2 = dice_bbs_2.aggregate()

board.add_scalar('Dice/Random Coord 0', dice_rc_0.mean().item(), 0)
board.add_scalar('Dice/Random Coord 1', dice_rc_1.mean().item(), 1)
board.add_scalar('Dice/Random Coord 2', dice_rc_2.mean().item(), 2) 
board.add_scalar('Dice/3 Random Splits 0', dice_rs3_0.mean().item(), 0)
board.add_scalar('Dice/3 Random Splits 1', dice_rs3_1.mean().item(), 1)
board.add_scalar('Dice/3 Random Splits 2', dice_rs3_2.mean().item(), 2)
board.add_scalar('Dice/5 Random Splits 0', dice_rs5_0.mean().item(), 0)
board.add_scalar('Dice/5 Random Splits 1', dice_rs5_1.mean().item(), 1)
board.add_scalar('Dice/5 Random Splits 2', dice_rs5_2.mean().item(), 2)
board.add_scalar('Dice/Central Point 0', dice_cp_0.mean().item(), 0)
board.add_scalar('Dice/Central Point 1', dice_cp_1.mean().item(), 1)
board.add_scalar('Dice/Central Point 2', dice_cp_2.mean().item(), 2)
board.add_scalar('Dice/Bounding Box 0', dice_bb_0.mean().item(), 0)
board.add_scalar('Dice/Bounding Box 1', dice_bb_1.mean().item(), 1)
board.add_scalar('Dice/Bounding Box 2', dice_bb_2.mean().item(), 2)
board.add_scalar('Dice/Bounding Box Similar 0', dice_bbs_0.mean().item(), 0)
board.add_scalar('Dice/Bounding Box Similar 1', dice_bbs_1.mean().item(), 1)
board.add_scalar('Dice/Bounding Box Similar 2', dice_bbs_2.mean().item(), 2)

board.add_scalar('Dice_simple/Random Coord', dice_rc_0.mean().item(), 0)
board.add_scalar('Dice_simple/Random Coord', dice_rc_1.mean().item(), 1)
board.add_scalar('Dice_simple/Random Coord', dice_rc_2.mean().item(), 2)
board.add_scalar('Dice_simple/3 Random Splits', dice_rs3_0.mean().item(), 0)
board.add_scalar('Dice_simple/3 Random Splits', dice_rs3_1.mean().item(), 1)
board.add_scalar('Dice_simple/3 Random Splits', dice_rs3_2.mean().item(), 2)
board.add_scalar('Dice_simple/5 Random Splits', dice_rs5_0.mean().item(), 0)
board.add_scalar('Dice_simple/5 Random Splits', dice_rs5_1.mean().item(), 1)
board.add_scalar('Dice_simple/5 Random Splits', dice_rs5_2.mean().item(), 2)
board.add_scalar('Dice_simple/Central Point', dice_cp_0.mean().item(), 0)
board.add_scalar('Dice_simple/Central Point', dice_cp_1.mean().item(), 1)
board.add_scalar('Dice_simple/Central Point', dice_cp_2.mean().item(), 2)
board.add_scalar('Dice_simple/Bounding Box', dice_bb_0.mean().item(), 0)
board.add_scalar('Dice_simple/Bounding Box', dice_bb_1.mean().item(), 1)
board.add_scalar('Dice_simple/Bounding Box', dice_bb_2.mean().item(), 2)
board.add_scalar('Dice_simple/Bounding Box Similar', dice_bbs_0.mean().item(), 0)
board.add_scalar('Dice_simple/Bounding Box Similar', dice_bbs_1.mean().item(), 1)
board.add_scalar('Dice_simple/Bounding Box Similar', dice_bbs_2.mean().item(), 2)


print('dice_rc_0', dice_rc_0.mean().item())
print('dice_rc_1', dice_rc_1.mean().item())
print('dice_rc_2', dice_rc_2.mean().item())
print('dice_rs3_0', dice_rs3_0.mean().item())
print('dice_rs3_1', dice_rs3_1.mean().item())
print('dice_rs3_2', dice_rs3_2.mean().item())
print('dice_rs5_0', dice_rs5_0.mean().item())
print('dice_rs5_1', dice_rs5_1.mean().item())
print('dice_rs5_2', dice_rs5_2.mean().item())
print('dice_cp_0', dice_cp_0.mean().item()) 
print('dice_cp_1', dice_cp_1.mean().item())
print('dice_cp_2', dice_cp_2.mean().item()) 
print('dice_bb_0', dice_bb_0.mean().item())
print('dice_bb_1', dice_bb_1.mean().item())
print('dice_bb_2', dice_bb_2.mean().item())
print('dice_bbs_0', dice_bbs_0.mean().item())
print('dice_bbs_1', dice_bbs_1.mean().item())
print('dice_bbs_2', dice_bbs_2.mean().item())

print('finished')

board.close()
