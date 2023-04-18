export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 1 \
--dataset mamo_US \
--name mamo_US_vit-h \
--variation 30 

python3 main_2d.py \
--model vit-b \
--gpu 1 \
--dataset mamo_US \
--name mamo_US_vit-b \
--variation 30 

python3 main_2d.py \
--model vit-l \
--gpu 1 \
--dataset mamo_US \
--name mamo_US_vit-l \
--variation 30 