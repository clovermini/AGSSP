_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '/XXX/datasets/collection_of_images/pretrain_distill_1017/'
class_name = ('anomaly')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette='random')

work_dir = './work_dirs/metal_distill_multi_frozen/yolov8_s_syncbn_fast_1xb32-10e_pretrain_frozen'

max_epochs = 10
train_batch_size_per_gpu = 200
train_num_workers = 8  

load_from = '/XXX/defect_detection/weights/own_pretrain/metal_v3_distill_yolov8s_cos_multi_frozen_epoch_200.pth'

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)))

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

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json', classwise=True)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=5, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=5), 
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=100,
        min_delta=0.005))
train_cfg = dict(max_epochs=max_epochs, val_interval=1)  

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]) 
randomness = dict(seed=2024, deterministic=True, diff_rank_seed=False)


