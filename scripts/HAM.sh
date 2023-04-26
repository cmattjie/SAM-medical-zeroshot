export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_2d.py \
--model vit-h \
--gpu 4 \
--dataset HAM \
--name HAM_vit-h \
--variation 30 

python3 main_2d.py \
--model vit-b \
--gpu 4 \
--dataset HAM \
--name HAM_vit-b \
--variation 30 

python3 main_2d.py \
--model vit-l \
--gpu 4 \
--dataset HAM \
--name HAM_vit-l \
--variation 30 