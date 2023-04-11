import numpy as np
import cv2

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

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
    mask = cv2.erode(mask, np.ones((50,50), np.uint8), iterations=1)
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # Obtém as coordenadas de todos os pixels com o valor desejado
    coordenadas = np.column_stack(np.where(mask == 1))

    # Seleciona uma coordenada aleatória
    coordinate = coordenadas[np.random.choice(len(coordenadas))]
    input_point = np.array([[coordinate[1], coordinate[0]]])
    return input_point, np.array([1])

def boundbox(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the bounding box of the first contour
    x, y, w, h = cv2.boundingRect(contours[0])
    # Create the bounding box coordinates
    bbox = [x, y, x+w, y+h]
    return np.array(bbox)

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