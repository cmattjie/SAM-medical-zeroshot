export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 0 \
--dataset hip \
--name hip_ilios_vit-h \
--variation 30 \
--hip_bone ilios

python3 main_2d.py \
--model vit-b \
--gpu 0 \
--dataset hip \
--name hip_ilios_vit-b \
--variation 30 \
--hip_bone ilios

python3 main_2d.py \
--model vit-l \
--gpu 0 \
--dataset hip \
--name hip_ilios_vit-l \
--variation 30 \
--hip_bone ilios