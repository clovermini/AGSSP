export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_VISIBLE_DEVICES=0

python -u tools/train.py configs/pretrain_agdp/yolov8_s_swinb_1xb32-10e_pretrain_frozen.py --amp
