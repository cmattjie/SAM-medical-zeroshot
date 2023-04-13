export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model default \
--gpu 2 \
--dataset ISIC \
--name ISIC