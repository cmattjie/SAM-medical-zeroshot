import glob
import os
import cv2
import torch
from tqdm import tqdm
import argparse
import csv

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

from segment_anything import sam_model_registry, SamPredictor
from monai.transforms import LoadImaged, ScaleIntensityRanged, Compose, Identity
from monai.data import DataLoader, Dataset, ThreadDataLoader
from monai.metrics import DiceMetric, MeanIoU
from utils import utils
import warnings
import json
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image

def get_args():
    parser = argparse.ArgumentParser(description='parameters for evaluating SAM')
    
    parser.add_argument('--model',              default='default', type=str, help='Model type for SAM')
    parser.add_argument('--dataset',            default='ISIC', type=str, help='dataset')
    parser.add_argument('--gpu',                default=2, type=str, help='GPU Number.')
    parser.add_argument('--name',               default='test', type=str, help='Run name on Tensorboard and savedirs.')
    parser.add_argument('--variation',          default='default', type=int, help='Variation of bounding box placement.')
    parser.add_argument('--hip_bone',           default='femur', type=str, help='hip bone to segment: femur or ilios')
    #parser.add_argument('--mask_mode',          default='rnd', type=str, help='Method for sampling points.')
    #parser.add_argument('--n_splits',           default=3, type=int, help='Number of splits to get points of.')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    return args 

args = get_args()
dicts = json.load(open('./utils/dicts.json', 'r'))
board = SummaryWriter(log_dir='./runs/'+args.name)

print(f"Name: {args.name}")
print(f"Model: {args.model}")
print(f"Dataset: {args.dataset}")

assert args.dataset in dicts['dataset_processed_path'].keys(), "Dataset not found"
assert args.model in dicts['sam_checkpoint'].keys(), "Model not found"

data_dir = dicts['dataset_processed_path'][args.dataset]
train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.*")))

hip_bone= args.hip_bone # 'femur' or 'ilios'
if args.dataset == 'hip':
    print(f"Segmenting {hip_bone}...")
    train_labels = sorted(glob.glob(os.path.join(data_dir, hip_bone, "*.*")))
else:
    train_labels = sorted(glob.glob(os.path.join(data_dir, "mask", "*.*")))

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files = data_dicts

check_ds = Dataset(data=train_files)# transform=train_transforms)
loader = DataLoader(check_ds, batch_size=1, shuffle=False)#, num_workers=4, shuffle=False)

sam_checkpoint = dicts['sam_checkpoint'][args.model]
device = torch.device(f'cuda:{args.gpu}')
print(device)
model_type = args.model
#change "-" to "_" to get the model name
model_type = model_type.replace("-", "_")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

dice_rc_0 = DiceMetric(include_background=True, reduction="none")
dice_rc_1 = DiceMetric(include_background=True, reduction="none")
dice_rc_2 = DiceMetric(include_background=True, reduction="none")
dice_rs3_0 = DiceMetric(include_background=True, reduction="none")
dice_rs3_1 = DiceMetric(include_background=True, reduction="none")
dice_rs3_2 = DiceMetric(include_background=True, reduction="none")
dice_rs5_0 = DiceMetric(include_background=True, reduction="none")
dice_rs5_1 = DiceMetric(include_background=True, reduction="none")
dice_rs5_2 = DiceMetric(include_background=True, reduction="none")
dice_cp_0 = DiceMetric(include_background=True, reduction="none")
dice_cp_1 = DiceMetric(include_background=True, reduction="none")
dice_cp_2 = DiceMetric(include_background=True, reduction="none")
dice_bb_0 = DiceMetric(include_background=True, reduction="none")
dice_bb_1 = DiceMetric(include_background=True, reduction="none")
dice_bb_2 = DiceMetric(include_background=True, reduction="none")
dice_bbs_05_0 = DiceMetric(include_background=True, reduction="none")
dice_bbs_05_1 = DiceMetric(include_background=True, reduction="none")
dice_bbs_05_2 = DiceMetric(include_background=True, reduction="none")
dice_bbs_1_0 = DiceMetric(include_background=True, reduction="none")
dice_bbs_1_1 = DiceMetric(include_background=True, reduction="none")
dice_bbs_1_2 = DiceMetric(include_background=True, reduction="none")
dice_bbs_2_0 = DiceMetric(include_background=True, reduction="none")
dice_bbs_2_1 = DiceMetric(include_background=True, reduction="none")
dice_bbs_2_2 = DiceMetric(include_background=True, reduction="none")

# iou_rc_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rc_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rc_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs3_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs3_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs3_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs5_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs5_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_rs5_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_cp_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_cp_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_cp_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bb_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bb_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bb_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_05_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_05_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_05_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_1_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_1_1 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_1_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_2_0 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
# iou_bbs_2_1 = MeanIoU(include_background=True, reduction="tmean", get_not_nans=False)
# iou_bbs_2_2 = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

#TODO ARRUMAR ERODE E ARRUMAR VARIAÇÃO BOUNDING BOX
#loop with tqdm
n_images = len(loader)

#get 10 random values from n_images
random_values = [1,2,3,4,5]#np.random.randint(0, n_images, 10)
#print(f"Random values: {random_values}")

#erode
erode=30
if args.dataset in ["mamo_US", "HAM", "hip"]:
    erode=10
if args.dataset in ['CVC']:
    erode=0
    
print('erode: ', erode)
df = pd.DataFrame(columns=['image', 'dice_rc_0', 'dice_rc_1', 'dice_rc_2', 'dice_rs3_0', 'dice_rs3_1', 'dice_rs3_2', 'dice_rs5_0', 'dice_rs5_1', 'dice_rs5_2', 'dice_cp_0', 'dice_cp_1', 'dice_cp_2', 'dice_bb_0', 'dice_bb_1', 'dice_bb_2', 'dice_bbs_05_0', 'dice_bbs_05_1', 'dice_bbs_05_2', 'dice_bbs_1_0', 'dice_bbs_1_1', 'dice_bbs_1_2', 'dice_bbs_2_0', 'dice_bbs_2_1', 'dice_bbs_2_2'])

for idx, batch in enumerate(tqdm(loader)):
    # if idx < 600:
    #    continue
    
    image_loc, mask_loc = batch["image"][0], batch["label"][0]

    #reading images and mask
    image = cv2.imread(image_loc, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_loc, cv2.IMREAD_GRAYSCALE)
    
    # #turn all masks to 1
    # mask[mask>0] = 1
    
    #FILL HOLES on mask
    if args.dataset == 'ISIC':
        contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(mask,[cnt],0,255,-1)
            
    #split mask
    masks_split = utils.split_mask(mask, args.dataset)
    
    if args.dataset == "CXRkaggle" and len(masks_split) !=2:
        print(f"only 1 lung mask found in image: {image_loc}")
        
    #create lists and set image
    masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs_05, masks_bbs_1, masks_bbs_2 = [], [], [], [], [], [], [], []
    
    predictor.set_image(image)
    
    for mask_s in masks_split:
        #get points and boxes
        try:
            input_rc, input_label_rc = utils.random_coordinate(mask_s, erode)
            input_rs3, input_label_rs3 = utils.random_splits(mask_s, n_splits=3, erode=erode)
            input_rs5, input_label_rs5 = utils.random_splits(mask_s, n_splits=5, erode=erode)
            input_cp, input_label_cp = utils.central_point(mask_s)
            input_bb, input_bbs_05 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.05)
            _, input_bbs_1 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.1)
            _, input_bbs_2 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.2)
     
            #predict masks from points and boxes
            masks_rc_temp, scores_rc, logits_rc = predictor.predict(point_coords=input_rc, point_labels=input_label_rc,multimask_output=True)
            masks_rs3_temp, scores_rs3, logits_rs3 = predictor.predict(point_coords=input_rs3, point_labels=input_label_rs3,multimask_output=True)
            masks_rs5_temp, scores_rs5, logits_rs5 = predictor.predict(point_coords=input_rs5, point_labels=input_label_rs5,multimask_output=True)
            masks_cp_temp, scores_cp, logits_cp = predictor.predict(point_coords=input_cp, point_labels=input_label_cp,multimask_output=True)
            masks_bb_temp, scores_bb, logits_bb = predictor.predict(point_coords=None, point_labels=None, box=input_bb, multimask_output=True)
            masks_bbs_05_temp, scores_bbs_05, logits_bbs_05 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_05, multimask_output=True)
            masks_bbs_1_temp, scores_bbs_1, logits_bbs_1 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_1, multimask_output=True)
            masks_bbs_2_temp, scores_bbs_2, logits_bbs_2 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_2, multimask_output=True)
            
            #append masks
            masks_rc.append(masks_rc_temp)
            masks_rs3.append(masks_rs3_temp)
            masks_rs5.append(masks_rs5_temp)
            masks_cp.append(masks_cp_temp)
            masks_bb.append(masks_bb_temp)
            masks_bbs_05.append(masks_bbs_05_temp)
            masks_bbs_1.append(masks_bbs_1_temp)
            masks_bbs_2.append(masks_bbs_2_temp)
            
        except:
            #TODO SAVE COMPRESSED VERSION OF IMAGE
            print(f"Error in image {idx}")
            print(mask_loc)
            print('number of submasks: ', len(masks_split))            
            
            input_rc, input_label_rc = utils.random_coordinate(mask_s, erode)
            input_rs3, input_label_rs3 = utils.random_splits(mask_s, n_splits=3, erode=erode)
            input_rs5, input_label_rs5 = utils.random_splits(mask_s, n_splits=5, erode=erode)
            input_cp, input_label_cp = utils.central_point(mask_s)
            input_bb, input_bbs_05 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.05)
            _, input_bbs_1 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.1)
            _, input_bbs_2 = utils.boundbox_similar(mask_s, mask_s.shape, max_var_percentage=0.2)
     
            #predict masks from points and boxes
            masks_rc_temp, scores_rc, logits_rc = predictor.predict(point_coords=input_rc, point_labels=input_label_rc,multimask_output=True)
            masks_rs3_temp, scores_rs3, logits_rs3 = predictor.predict(point_coords=input_rs3, point_labels=input_label_rs3,multimask_output=True)
            masks_rs5_temp, scores_rs5, logits_rs5 = predictor.predict(point_coords=input_rs5, point_labels=input_label_rs5,multimask_output=True)
            masks_cp_temp, scores_cp, logits_cp = predictor.predict(point_coords=input_cp, point_labels=input_label_cp,multimask_output=True)
            masks_bb_temp, scores_bb, logits_bb = predictor.predict(point_coords=None, point_labels=None, box=input_bb, multimask_output=True)
            masks_bbs_05_temp, scores_bbs_05, logits_bbs_05 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_05, multimask_output=True)
            masks_bbs_1_temp, scores_bbs_1, logits_bbs_1 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_1, multimask_output=True)
            masks_bbs_2_temp, scores_bbs_2, logits_bbs_2 = predictor.predict(point_coords=None, point_labels=None, box=input_bbs_2, multimask_output=True)
            
            #append masks
            masks_rc.append(masks_rc_temp)
            masks_rs3.append(masks_rs3_temp)
            masks_rs5.append(masks_rs5_temp)
            masks_cp.append(masks_cp_temp)
            masks_bb.append(masks_bb_temp)
            masks_bbs_05.append(masks_bbs_05_temp)
            masks_bbs_1.append(masks_bbs_1_temp)
            masks_bbs_2.append(masks_bbs_2_temp)
            
    #lista (5 elementos) de listas (2 elementos) de listas de 3 predições de máscara
    mask_list = [masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs_05, masks_bbs_1, masks_bbs_2]
    masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs_05, masks_bbs_1, masks_bbs_2 = utils.merge_masks(mask_list)
    
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
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_bbs_05', np.expand_dims(masks_bbs_05*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_bbs_1', np.expand_dims(masks_bbs_1*1, axis=-1), 0, dataformats='NHWC')
        board.add_images(f'images_{mask_loc.split("/")[-1]}/mask_bbs_2', np.expand_dims(masks_bbs_2*1, axis=-1), 0, dataformats='NHWC')
    
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
    dice_bbs_05_0(y_pred=torch.Tensor(masks_bbs_05[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_05_1(y_pred=torch.Tensor((masks_bbs_05[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_05_2(y_pred=torch.Tensor((masks_bbs_05[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_1_0(y_pred=torch.Tensor(masks_bbs_1[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_1_1(y_pred=torch.Tensor((masks_bbs_1[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_1_2(y_pred=torch.Tensor((masks_bbs_1[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_2_0(y_pred=torch.Tensor(masks_bbs_2[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_2_1(y_pred=torch.Tensor((masks_bbs_2[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    dice_bbs_2_2(y_pred=torch.Tensor((masks_bbs_2[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    
    
    
    # iou_rc_0(y_pred=torch.Tensor(masks_rc[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rc_1(y_pred=torch.Tensor((masks_rc[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rc_2(y_pred=torch.Tensor((masks_rc[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs3_0(y_pred=torch.Tensor(masks_rs3[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs3_1(y_pred=torch.Tensor((masks_rs3[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs3_2(y_pred=torch.Tensor((masks_rs3[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs5_0(y_pred=torch.Tensor(masks_rs5[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs5_1(y_pred=torch.Tensor((masks_rs5[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_rs5_2(y_pred=torch.Tensor((masks_rs5[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_cp_0(y_pred=torch.Tensor(masks_cp[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_cp_1(y_pred=torch.Tensor((masks_cp[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_cp_2(y_pred=torch.Tensor((masks_cp[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bb_0(y_pred=torch.Tensor(masks_bb[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bb_1(y_pred=torch.Tensor((masks_bb[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bb_2(y_pred=torch.Tensor((masks_bb[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_05_0(y_pred=torch.Tensor(masks_bbs_05[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_05_1(y_pred=torch.Tensor((masks_bbs_05[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_05_2(y_pred=torch.Tensor((masks_bbs_05[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_1_0(y_pred=torch.Tensor(masks_bbs_1[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_1_1(y_pred=torch.Tensor((masks_bbs_1[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_1_2(y_pred=torch.Tensor((masks_bbs_1[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_2_0(y_pred=torch.Tensor(masks_bbs_2[0,:,:]*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_2_1(y_pred=torch.Tensor((masks_bbs_2[1,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0))
    # iou_bbs_2_2(y_pred=torch.Tensor((masks_bbs_2[2,:,:])*1).unsqueeze(0).unsqueeze(0), y=torch.Tensor(mask).unsqueeze(0).unsqueeze(0)) 

#before aggregation, save the dice values to the dataframe 'df'
#print(dice_rc_0.aggregate())
df['dice_rc_0'] = dice_rc_0.aggregate().tolist()
df['dice_rc_1'] = dice_rc_1.aggregate().tolist()
df['dice_rc_2'] = dice_rc_2.aggregate().tolist()
df['dice_rs3_0'] = dice_rs3_0.aggregate().tolist()
df['dice_rs3_1'] = dice_rs3_1.aggregate().tolist()
df['dice_rs3_2'] = dice_rs3_2.aggregate().tolist()
df['dice_rs5_0'] = dice_rs5_0.aggregate().tolist()
df['dice_rs5_1'] = dice_rs5_1.aggregate().tolist()
df['dice_rs5_2'] = dice_rs5_2.aggregate().tolist()
df['dice_cp_0'] = dice_cp_0.aggregate().tolist()
df['dice_cp_1'] = dice_cp_1.aggregate().tolist()
df['dice_cp_2'] = dice_cp_2.aggregate().tolist()
df['dice_bb_0'] = dice_bb_0.aggregate().tolist()
df['dice_bb_1'] = dice_bb_1.aggregate().tolist()
df['dice_bb_2'] = dice_bb_2.aggregate().tolist()
df['dice_bbs_05_0'] = dice_bbs_05_0.aggregate().tolist()
df['dice_bbs_05_1'] = dice_bbs_05_1.aggregate().tolist()
df['dice_bbs_05_2'] = dice_bbs_05_2.aggregate().tolist()
df['dice_bbs_1_0'] = dice_bbs_1_0.aggregate().tolist()
df['dice_bbs_1_1'] = dice_bbs_1_1.aggregate().tolist()
df['dice_bbs_1_2'] = dice_bbs_1_2.aggregate().tolist()
df['dice_bbs_2_0'] = dice_bbs_2_0.aggregate().tolist()
df['dice_bbs_2_1'] = dice_bbs_2_1.aggregate().tolist()
df['dice_bbs_2_2'] = dice_bbs_2_2.aggregate().tolist()

#saving all dices to a csv file
if args.dataset == "hip":
    df.to_csv(f'results_paper/dice_values{args.dataset}{hip_bone}_{args.model}.csv')
else:
    df.to_csv(f'results_paper/dice_values{args.dataset}_{args.model}.csv')
    
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
dice_bbs_05_0 = dice_bbs_05_0.aggregate()
dice_bbs_05_1 = dice_bbs_05_1.aggregate()
dice_bbs_05_2 = dice_bbs_05_2.aggregate()
dice_bbs_1_0 = dice_bbs_1_0.aggregate()
dice_bbs_1_1 = dice_bbs_1_1.aggregate()
dice_bbs_1_2 = dice_bbs_1_2.aggregate()
dice_bbs_2_0 = dice_bbs_2_0.aggregate()
dice_bbs_2_1 = dice_bbs_2_1.aggregate()
dice_bbs_2_2 = dice_bbs_2_2.aggregate()

# iou_rc_0 = iou_rc_0.aggregate()
# iou_rc_1 = iou_rc_1.aggregate()
# iou_rc_2 = iou_rc_2.aggregate()
# iou_rs3_0 = iou_rs3_0.aggregate()
# iou_rs3_1 = iou_rs3_1.aggregate()
# iou_rs3_2 = iou_rs3_2.aggregate()
# iou_rs5_0 = iou_rs5_0.aggregate()
# iou_rs5_1 = iou_rs5_1.aggregate()
# iou_rs5_2 = iou_rs5_2.aggregate()
# iou_cp_0 = iou_cp_0.aggregate()
# iou_cp_1 = iou_cp_1.aggregate()
# iou_cp_2 = iou_cp_2.aggregate()
# iou_bb_0 = iou_bb_0.aggregate()
# iou_bb_1 = iou_bb_1.aggregate()
# iou_bb_2 = iou_bb_2.aggregate()
# iou_bbs_05_0 = iou_bbs_05_0.aggregate()
# iou_bbs_05_1 = iou_bbs_05_1.aggregate()
# iou_bbs_05_2 = iou_bbs_05_2.aggregate()
# iou_bbs_1_0 = iou_bbs_1_0.aggregate()
# iou_bbs_1_1 = iou_bbs_1_1.aggregate()
# iou_bbs_1_2 = iou_bbs_1_2.aggregate()
# iou_bbs_2_0 = iou_bbs_2_0.aggregate()
# iou_bbs_2_1 = iou_bbs_2_1.aggregate()
# iou_bbs_2_2 = iou_bbs_2_2.aggregate()

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
board.add_scalar('Dice/Bounding Box Similar 0.05 0', dice_bbs_05_0.mean().item(), 0)
board.add_scalar('Dice/Bounding Box Similar 0.05 1', dice_bbs_05_1.mean().item(), 1)
board.add_scalar('Dice/Bounding Box Similar 0.05 2', dice_bbs_05_2.mean().item(), 2)
board.add_scalar('Dice/Bounding Box Similar 0.1 0', dice_bbs_1_0.mean().item(), 0)
board.add_scalar('Dice/Bounding Box Similar 0.1 1', dice_bbs_1_1.mean().item(), 1)
board.add_scalar('Dice/Bounding Box Similar 0.1 2', dice_bbs_1_2.mean().item(), 2)
board.add_scalar('Dice/Bounding Box Similar 0.2 0', dice_bbs_2_0.mean().item(), 0)
board.add_scalar('Dice/Bounding Box Similar 0.2 1', dice_bbs_2_1.mean().item(), 1)
board.add_scalar('Dice/Bounding Box Similar 0.2 2', dice_bbs_2_2.mean().item(), 2)

# board.add_scalar('iou/Random Coord 0', iou_rc_0.mean().item(), 0)
# board.add_scalar('iou/Random Coord 1', iou_rc_1.mean().item(), 1)
# board.add_scalar('iou/Random Coord 2', iou_rc_2.mean().item(), 2) 
# board.add_scalar('iou/3 Random Splits 0', iou_rs3_0.mean().item(), 0)
# board.add_scalar('iou/3 Random Splits 1', iou_rs3_1.mean().item(), 1)
# board.add_scalar('iou/3 Random Splits 2', iou_rs3_2.mean().item(), 2)
# board.add_scalar('iou/5 Random Splits 0', iou_rs5_0.mean().item(), 0)
# board.add_scalar('iou/5 Random Splits 1', iou_rs5_1.mean().item(), 1)
# board.add_scalar('iou/5 Random Splits 2', iou_rs5_2.mean().item(), 2)
# board.add_scalar('iou/Central Point 0', iou_cp_0.mean().item(), 0)
# board.add_scalar('iou/Central Point 1', iou_cp_1.mean().item(), 1)
# board.add_scalar('iou/Central Point 2', iou_cp_2.mean().item(), 2)
# board.add_scalar('iou/Bounding Box 0', iou_bb_0.mean().item(), 0)
# board.add_scalar('iou/Bounding Box 1', iou_bb_1.mean().item(), 1)
# board.add_scalar('iou/Bounding Box 2', iou_bb_2.mean().item(), 2)
# board.add_scalar('iou/Bounding Box Similar 0.05 0', iou_bbs_05_0.mean().item(), 0)
# board.add_scalar('iou/Bounding Box Similar 0.05 1', iou_bbs_05_1.mean().item(), 1)
# board.add_scalar('iou/Bounding Box Similar 0.05 2', iou_bbs_05_2.mean().item(), 2)
# board.add_scalar('iou/Bounding Box Similar 0.1 0', iou_bbs_1_0.mean().item(), 0)
# board.add_scalar('iou/Bounding Box Similar 0.1 1', iou_bbs_1_1.mean().item(), 1)
# board.add_scalar('iou/Bounding Box Similar 0.1 2', iou_bbs_1_2.mean().item(), 2)
# board.add_scalar('iou/Bounding Box Similar 0.2 0', iou_bbs_2_0.mean().item(), 0)
# board.add_scalar('iou/Bounding Box Similar 0.2 1', iou_bbs_2_1.mean().item(), 1)
# board.add_scalar('iou/Bounding Box Similar 0.2 2', iou_bbs_2_2.mean().item(), 2)

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
board.add_scalar('Dice_simple/Bounding Box Similar 0.05', dice_bbs_05_0.mean().item(), 0)
board.add_scalar('Dice_simple/Bounding Box Similar 0.05', dice_bbs_05_1.mean().item(), 1)
board.add_scalar('Dice_simple/Bounding Box Similar 0.05', dice_bbs_05_2.mean().item(), 2)
board.add_scalar('Dice_simple/Bounding Box Similar 0.1', dice_bbs_1_0.mean().item(), 0)
board.add_scalar('Dice_simple/Bounding Box Similar 0.1', dice_bbs_1_1.mean().item(), 1)
board.add_scalar('Dice_simple/Bounding Box Similar 0.1', dice_bbs_1_2.mean().item(), 2)
board.add_scalar('Dice_simple/Bounding Box Similar 0.2', dice_bbs_2_0.mean().item(), 0)
board.add_scalar('Dice_simple/Bounding Box Similar 0.2', dice_bbs_2_1.mean().item(), 1)
board.add_scalar('Dice_simple/Bounding Box Similar 0.2', dice_bbs_2_2.mean().item(), 2)

# board.add_scalar('iou_simple/Random Coord', iou_rc_0.mean().item(), 0)
# board.add_scalar('iou_simple/Random Coord', iou_rc_1.mean().item(), 1)
# board.add_scalar('iou_simple/Random Coord', iou_rc_2.mean().item(), 2)
# board.add_scalar('iou_simple/3 Random Splits', iou_rs3_0.mean().item(), 0)
# board.add_scalar('iou_simple/3 Random Splits', iou_rs3_1.mean().item(), 1)
# board.add_scalar('iou_simple/3 Random Splits', iou_rs3_2.mean().item(), 2)
# board.add_scalar('iou_simple/5 Random Splits', iou_rs5_0.mean().item(), 0)
# board.add_scalar('iou_simple/5 Random Splits', iou_rs5_1.mean().item(), 1)
# board.add_scalar('iou_simple/5 Random Splits', iou_rs5_2.mean().item(), 2)
# board.add_scalar('iou_simple/Central Point', iou_cp_0.mean().item(), 0)
# board.add_scalar('iou_simple/Central Point', iou_cp_1.mean().item(), 1)
# board.add_scalar('iou_simple/Central Point', iou_cp_2.mean().item(), 2)
# board.add_scalar('iou_simple/Bounding Box', iou_bb_0.mean().item(), 0)
# board.add_scalar('iou_simple/Bounding Box', iou_bb_1.mean().item(), 1)
# board.add_scalar('iou_simple/Bounding Box', iou_bb_2.mean().item(), 2)
# board.add_scalar('iou_simple/Bounding Box Similar 0.05', iou_bbs_05_0.mean().item(), 0)
# board.add_scalar('iou_simple/Bounding Box Similar 0.05', iou_bbs_05_1.mean().item(), 1)
# board.add_scalar('iou_simple/Bounding Box Similar 0.05', iou_bbs_05_2.mean().item(), 2)
# board.add_scalar('iou_simple/Bounding Box Similar 0.1', iou_bbs_1_0.mean().item(), 0)
# board.add_scalar('iou_simple/Bounding Box Similar 0.1', iou_bbs_1_1.mean().item(), 1)
# board.add_scalar('iou_simple/Bounding Box Similar 0.1', iou_bbs_1_2.mean().item(), 2)
# board.add_scalar('iou_simple/Bounding Box Similar 0.2', iou_bbs_2_0.mean().item(), 0)
# board.add_scalar('iou_simple/Bounding Box Similar 0.2', iou_bbs_2_1.mean().item(), 1)
# board.add_scalar('iou_simple/Bounding Box Similar 0.2', iou_bbs_2_2.mean().item(), 2)

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
print('dice_bbs_05_0', dice_bbs_05_0.mean().item())
print('dice_bbs_05_1', dice_bbs_05_1.mean().item())
print('dice_bbs_05_2', dice_bbs_05_2.mean().item())
print('dice_bbs_1_0', dice_bbs_1_0.mean().item())
print('dice_bbs_1_1', dice_bbs_1_1.mean().item())
print('dice_bbs_1_2', dice_bbs_1_2.mean().item())
print('dice_bbs_2_0', dice_bbs_2_0.mean().item())
print('dice_bbs_2_1', dice_bbs_2_1.mean().item())
print('dice_bbs_2_2', dice_bbs_2_2.mean().item())

# #add to csv
# with open(f'{args.dataset}_{args.model}.csv', 'a') as f:
#     writer = csv.writer(f)
#     #create header
#     if os.stat(f'{args.dataset}_{args.model}.csv').st_size == 0:
#         writer.writerow(['dice_rc_0', 'dice_rc_1', 'dice_rc_2', 'dice_rs3_0', 'dice_rs3_1', 'dice_rs3_2', 'dice_rs5_0', 'dice_rs5_1', 'dice_rs5_2', 'dice_cp_0', 'dice_cp_1', 'dice_cp_2', 'dice_bb_0', 'dice_bb_1', 'dice_bb_2', 'dice_bbs_05_0', 'dice_bbs_05_1', 'dice_bbs_05_2', 'dice_bbs_1_0', 'dice_bbs_1_1', 'dice_bbs_1_2', 'dice_bbs_2_0', 'dice_bbs_2_1', 'dice_bbs_2_2',
#                         #  'iou_rc_0', 'iou_rc_1', 'iou_rc_2', 'iou_rs3_0', 'iou_rs3_1', 'iou_rs3_2', 'iou_rs5_0', 'iou_rs5_1', 'iou_rs5_2', 'iou_cp_0', 'iou_cp_1', 'iou_cp_2', 'iou_bb_0', 'iou_bb_1', 'iou_bb_2', 'iou_bbs_05_0', 'iou_bbs_05_1', 'iou_bbs_05_2', 'iou_bbs_1_0', 'iou_bbs_1_1', 'iou_bbs_1_2', 'iou_bbs_2_0', 'iou_bbs_2_1', 'iou_bbs_2_2'
#                         ])
#     writer.writerow([dice_rc_0.mean().item(), dice_rc_1.mean().item(), dice_rc_2.mean().item(), dice_rs3_0.mean().item(), dice_rs3_1.mean().item(), dice_rs3_2.mean().item(), dice_rs5_0.mean().item(), dice_rs5_1.mean().item(), dice_rs5_2.mean().item(), dice_cp_0.mean().item(), dice_cp_1.mean().item(), dice_cp_2.mean().item(), dice_bb_0.mean().item(), dice_bb_1.mean().item(), dice_bb_2.mean().item(), dice_bbs_05_0.mean().item(), dice_bbs_05_1.mean().item(), dice_bbs_05_2.mean().item(), dice_bbs_1_0.mean().item(), dice_bbs_1_1.mean().item(), dice_bbs_1_2.mean().item(), dice_bbs_2_0.mean().item(), dice_bbs_2_1.mean().item(), dice_bbs_2_2.mean().item(),
#                     # iou_rc_0.mean().item(), iou_rc_1.mean().item(), iou_rc_2.mean().item(), iou_rs3_0.mean().item(), iou_rs3_1.mean().item(), iou_rs3_2.mean().item(), iou_rs5_0.mean().item(), iou_rs5_1.mean().item(), iou_rs5_2.mean().item(), iou_cp_0.mean().item(), iou_cp_1.mean().item(), iou_cp_2.mean().item(), iou_bb_0.mean().item(), iou_bb_1.mean().item(), iou_bb_2.mean().item(), iou_bbs_05_0.mean().item(), iou_bbs_05_1.mean().item(), iou_bbs_05_2.mean().item(), iou_bbs_1_0.mean().item(), iou_bbs_1_1.mean().item(), iou_bbs_1_2.mean().item(), iou_bbs_2_0.mean().item(), iou_bbs_2_1.mean().item(), iou_bbs_2_2.mean().item()
#                     ])

print('finished')
board.close()
