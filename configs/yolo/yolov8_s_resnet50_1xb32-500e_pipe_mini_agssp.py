_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '/home/data/Datasets/public/pipeData/'
class_name = ('warp', 'external fold', 'wrinkle', 'scratch')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette='random')   
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

work_dir = './work_dirs/own_pretrain/metal_ft_cos_frozen/train_mini_500/yolov8_s_resnet50_1xb32-500e_pipe_4cate/' 

max_epochs = 500
train_batch_size_per_gpu = 32
train_num_workers = 4  

base_lr = _base_.base_lr / 4   
optim_wrapper = dict(optimizer=dict(lr=base_lr))  

load_from = '/home/data/XXX/projects/defect_detection/weights/own_pretrain/metal_ft_rs50_cos_frozen_epoch_10.pth'

channels = [512, 1024, 2048]
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
model = dict(
    backbone=dict(
        _delete_=True,  # Delete the backbone field in _base_
        type='mmdet.ResNet',  # Using ResNet from mmdet
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),  # [256, 512, 1024, 2048]  yolov8fpn [256, 512, 1024]
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        ), 
    neck=dict(
        type='YOLOv8PAFPN',
        widen_factor=widen_factor,
        in_channels=channels, # Note: The 3 channels of ResNet-50 output are [512, 1024, 2048], which do not match the original yolov5-s neck and need to be changed.
        out_channels=channels),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=channels, # input channels of head need to be changed accordingly
            widen_factor=widen_factor)),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_mini_500_4cate.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val_4cate.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val_4cate.json', classwise=True)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='coco/bbox_mAP_50'),
    # The warmup_mim_iter parameter is critical.
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=5), 
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=100,
        min_delta=0.005))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)  

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])  # tensorboard 

randomness = dict(seed=2024, deterministic=True, diff_rank_seed=False)


