import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from utils import utils

# Carrega a imagem
img = cv2.imread('/mnt/B-SSD/maltamed/datasets/2D/ISIC/images/ISIC_0010231.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Cria uma janela raiz para o diálogo de seleção de arquivo
#root = tk.Tk()
#root.withdraw()
# Abre o diálogo de seleção de arquivo para o usuário selecionar uma imagem
#file_path = filedialog.askopenfilename()
#img = plt.imread(file_path)

sam_checkpoint = "/mnt/B-SSD/maltamed/SAM-zero-shot/segment-anything-main/checkpoints/sam_vit_h.pth"
device = "cuda"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(img)

# variáveis globais
top_left = None
bottom_right = None
drawing = False

# função de callback para o evento de clique do mouse
def mouse_callback(event, x, y, flags, params):
    global top_left, bottom_right, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        top_left = (x, y)
        drawing = True
    
    elif event == cv2.EVENT_LBUTTONUP:
        bottom_right = (x, y)
        drawing = False

# cria uma cópia da imagem para desenhar a bounding box
draw = img.copy()

# exibe a imagem e a mensagem em uma janela
cv2.imshow('image', draw)
cv2.putText(draw, 'Desenhe a bounding box, e quando for aceitavel, pressione Enter para continuar', (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# define a função de callback para o evento de clique do mouse
cv2.setMouseCallback('image', mouse_callback)

# aguarda até que uma bounding box seja desenhada e a tecla Enter seja pressionada
while True:
    # atualiza a imagem desenhando a bounding box
    draw = img.copy()
    if top_left is not None and bottom_right is not None:
        cv2.rectangle(draw, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(draw, 'Desenhe a bounding box, e quando for aceitavel, pressione Enter para continuar', (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow('image', draw)
    
    # aguarda por uma tecla
    key = cv2.waitKey(1) & 0xFF
    
    # se a tecla 'c' for pressionada, retorna as coordenadas da bounding box
    if key == ord('c') and top_left is not None and bottom_right is not None:
        print(f'Top left: {top_left}, Bottom right: {bottom_right}')
        break
    
    # se a tecla Enter for pressionada, continua a execução do código
    elif key == 13:
        break

# fecha a janela
cv2.destroyAllWindows()

# Infere com o SAM
input_point = None
input_label = None
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=np.array(top_left + bottom_right),
    multimask_output=True,
)

