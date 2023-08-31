export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 1 \
--dataset hip \
--name hip_femur_vit-h \
--variation 30 \
--hip_bone femur

python3 main_2d.py \
--model vit-b \
--gpu 1 \
--dataset hip \
--name hip_femur_vit-b \
--variation 30 \
--hip_bone femur

python3 main_2d.py \
--model vit-l \
--gpu 1 \
--dataset hip \
--name hip_femur_vit-l \
--variation 30 \
--hip_bone femur