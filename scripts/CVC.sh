export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 2 \
--dataset CVC \
--name CVC_vit-h \
--variation 30 

python3 main_2d.py \
--model vit-b \
--gpu 2 \
--dataset CVC \
--name CVC_vit-b \
--variation 30 

python3 main_2d.py \
--model vit-l \
--gpu 2 \
--dataset CVC \
--name CVC_vit-l \
--variation 30 