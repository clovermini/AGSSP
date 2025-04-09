export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_VISIBLE_DEVICES=0

python -u tools/train.py configs/yolo/yolov8_s_syncbn_fast_1xb32-500e_casting_billet_mini_agssp.py --amp