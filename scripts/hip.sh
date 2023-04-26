export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 4 \
--dataset hip \
--name hip_vit-h \
--variation 30 

python3 main_2d.py \
--model vit-b \
--gpu 4 \
--dataset hip \
--name hip_vit-b \
--variation 30 

python3 main_2d.py \
--model vit-l \
--gpu 4 \
--dataset hip \
--name hip_vit-l \
--variation 30 