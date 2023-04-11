export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model default
--gpu 1 \
--dataset ISIC \
--name ISIC