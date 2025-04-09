CONFIG=configs/pretrain_agbp/simmim/simmim_swin-base-w6_1xb256-amp-coslr-500e_in1k-192px_metal_distill_multi.py
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CUBLAS_WORKSPACE_CONFIG=:16:8 python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        tools/train.py \
        $CONFIG \
	    --amp \
        --launcher pytorch ${@:3} \
        #--resume \

