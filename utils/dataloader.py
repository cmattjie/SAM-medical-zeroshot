from torch.utils.data import Dataset
import numpy as np
from utils import utils
import cv2

class CustomDataset(Dataset):
    def __init__(self, path_list, predictor, transform):
        self.path_list = path_list
        self.predictor = predictor
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        batch = self.transform(self.path_list[index])
        image, mask = batch["image"], batch["label"]
        
        mask = mask.numpy()

        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        image = np.array(image).astype(np.uint8)

        coordinate = utils.random_coordinate(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        #image = cv2.cvtColor(image)

        self.predictor.set_image(image)
        input_point = np.array([[coordinate[1], coordinate[0]]])
        input_label = np.array([1])
        masks_pred, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return masks_pred, mask