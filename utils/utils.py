import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage

def random_splits(mask, n_splits=3):
    mask = cv2.erode(mask, np.ones((30,30), np.uint8), iterations=1)
    # Find the indices of the rows where there are non-zero values
    non_zero_rows = np.where(np.any(mask, axis=1))[0]

    # split in three parts
    total = non_zero_rows[-1] - non_zero_rows[0]
    split_size = total // n_splits

    coords = []

    for n in range(n_splits):
        if n == 0:
            start = 0
            end = (non_zero_rows[0] + split_size * (n+1))

        elif n == n_splits-1:
            start = (non_zero_rows[0] + split_size * (n))+1 if n ==1 else (end+1)
            end = mask.shape[0]

        else:
            # Define a parte da imagem que deseja considerar
            start = (non_zero_rows[0] + split_size * (n))+1 if n ==1 else (end+1)
            end = (non_zero_rows[0] + split_size * (n+1))


        parte_desejada = mask[start:end, :] # por exemplo, uma região retangular da imagem

        # Obtém as coordenadas de todos os pixels com o valor desejado dentro da parte desejada
        coordenadas_parte = np.column_stack(np.where(parte_desejada > 0))

        # Seleciona uma coordenada aleatória da parte desejada
        coordenada_aleatoria_parte = coordenadas_parte[np.random.choice(len(coordenadas_parte))]

        # Converte a coordenada para as coordenadas da imagem completa
        coords.append([coordenada_aleatoria_parte[1], coordenada_aleatoria_parte[0]+start])

    input_point = np.array(coords)
    input_label = np.array([1]*n_splits)
    return input_point, input_label

def central_point(mask):
    #calcula o ponto central da máscara
    x_indices = np.where(np.sum(mask, axis=0) > 0)[0]
    y_indices = np.where(np.sum(mask, axis=1) > 0)[0]
    x_middle = int(np.mean(x_indices))
    y_middle = int(np.mean(y_indices))
    #retorna y antes de x por padronização
    input_point = np.array([[x_middle, y_middle]])
    input_label = np.array([1])
    return input_point, input_label

def random_coordinate(mask):
    mask = cv2.erode(mask, np.ones((30,30), np.uint8), iterations=1)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Obtém as coordenadas de todos os pixels com o valor desejado
    coordenadas = np.column_stack(np.where(mask == 1))

    # Seleciona uma coordenada aleatória
    coordinate = coordenadas[np.random.choice(len(coordenadas))]
    input_point = np.array([[coordinate[1], coordinate[0]]])
    return input_point, np.array([1])

def merge_masks(mask_list):
    masks_rc_list, masks_rs3_list, masks_rs5_list, masks_cp_list, masks_bb_list, masks_bbs_list = mask_list
    masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs = [], [], [], [], [], []
    if len(masks_rc_list) > 1:
        for i in range(len(masks_rc_list[0])): #i=3
            masks_rc.append(np.logical_or(masks_rc_list[0][i], masks_rc_list[1][i]))
            masks_rs3.append(np.logical_or(masks_rs3_list[0][i], masks_rs3_list[1][i]))
            masks_rs5.append(np.logical_or(masks_rs5_list[0][i], masks_rs5_list[1][i]))
            masks_cp.append(np.logical_or(masks_cp_list[0][i], masks_cp_list[1][i]))
            masks_bb.append(np.logical_or(masks_bb_list[0][i], masks_bb_list[1][i]))
            masks_bbs.append(np.logical_or(masks_bbs_list[0][i], masks_bbs_list[1][i]))
            
        masks_rc=np.stack(masks_rc, axis=0)
        masks_rs3=np.stack(masks_rs3, axis=0)
        masks_rs5=np.stack(masks_rs5, axis=0)
        masks_cp=np.stack(masks_cp, axis=0)
        masks_bb=np.stack(masks_bb, axis=0)
        masks_bbs=np.stack(masks_bbs, axis=0)

    else:
        masks_rc = masks_rc_list[0]
        masks_rs3 = masks_rs3_list[0]
        masks_rs5 = masks_rs5_list[0]
        masks_cp = masks_cp_list[0]
        masks_bb = masks_bb_list[0]
        masks_bbs = masks_bbs_list[0]
        
    return masks_rc, masks_rs3, masks_rs5, masks_cp, masks_bb, masks_bbs

def boundbox(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the bounding box of the first contour
    x, y, w, h = cv2.boundingRect(contours[0])
    # Create the bounding box coordinates
    bbox = [x, y, x+w, y+h]
    return np.array(bbox)

def split_mask(mask, dataset):
    mask = np.array(mask)
    if dataset=='ISIC':
        return [cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)[:,:,0]]
    labels, nlabels = ndimage.measurements.label(mask)
    if nlabels < 2:
        return [mask]
    # Calculate the size of each connected region
    region_sizes = [np.sum(labels == i) for i in range(1, nlabels + 1)]
    # Sort the indices of the regions by size in descending order and take the two largest
    largest_indices = np.argsort(region_sizes)[-2:][::-1] + 1
    masks = []
    for i in largest_indices:
        temp_mask = (labels == i) * 255
        temp_mask = cv2.cvtColor(temp_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)[:,:,0]
        masks.append(temp_mask)
    return masks

def boundbox_similar(mask, img_shape, pos_variation=10, size_variation=10, size_mode="random"):    
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the bounding box of the first contour
    x, y, w, h = cv2.boundingRect(contours[0])
    
    # Create the bounding box coordinates
    bbox = [x, y, x+w, y+h]
    
    # Define the maximum and minimum possible variations for position
    max_pos_variation = min(pos_variation, x, y, img_shape[1] - (x+w), img_shape[0] - (y+h))
    
    # Add random variation to the position of the bounding box's upper left corner
    new_x = np.clip(x + np.random.randint(-max_pos_variation, max_pos_variation+1), 0, img_shape[1]-w)
    new_y = np.clip(y + np.random.randint(-max_pos_variation, max_pos_variation+1), 0, img_shape[0]-h)
    
    # Calculate the size variation based on the size_mode parameter
    if size_mode == "bigger":
        size_var = np.random.randint(0, size_variation+1)
    elif size_mode == "smaller":
        size_var = np.random.randint(-size_variation, 1)
    else:  # Default to "random" if an unsupported mode is provided
        size_var = np.random.randint(-size_variation, size_variation+1)
    
    # Calculate the new width and height with the size variation
    new_w = np.clip(w + size_var, 1, img_shape[1] - new_x)
    new_h = np.clip(h + size_var, 1, img_shape[0] - new_y)
    
    # Create the similar_bbox coordinates
    similar_bbox = [new_x, new_y, new_x + new_w, new_y + new_h]
    
    return np.array(bbox), np.array(similar_bbox)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))