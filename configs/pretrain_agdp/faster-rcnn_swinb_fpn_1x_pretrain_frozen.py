_base_ = [
    './_base_/models/faster-rcnn_r50_fpn.py',
    './_base_/datasets/pretrain_dataset.py',
    './_base_/schedules/schedule_10e.py', './_base_/default_runtime.py'
]

data_root = '/XXX/datasets/collection_of_images/pretrain_distill_1017/'
class_name = ('anomaly')
num_classes = 1 # len(class_name)
metainfo = dict(classes=class_name, palette='random')  
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)


work_dir = './work_dirs/metal_simmim_distill_multi_frozen/faster-rcnn_swinb_fpn_1x_pretrain_frozen' 

max_epochs = 10
train_batch_size_per_gpu = 128
train_num_workers = 8  

load_from = '/XXX/defect_detection/weights/own_pretrain/metal_simmim_distill_multi_frozen_epoch_200.pth'
find_unused_parameters=True

model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[101.57, 101.57, 101.57], 
        std=[61.2, 61.2, 61.2]),
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmdet.SwinTransformer', # Using SwinTransformer from mmdet
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=6,  # 6 7 defalut
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=4,
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
        ),   # , prefix='backbone.'
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

#_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(classwise=False)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=10, save_best='coco/bbox_mAP_50',type='CheckpointHook'),
    logger=dict(type='LoggerHook', interval=5), 
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=100,
        min_delta=0.005))


#max_epochs=1000
train_cfg = dict(max_epochs=max_epochs, val_interval=10)  


# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]) 
randomness = dict(seed=2024, deterministic=False, diff_rank_seed=False)



