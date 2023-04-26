export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-b \
--gpu 2 \
--dataset CVC \
--name CVC_test \
--variation 30 
