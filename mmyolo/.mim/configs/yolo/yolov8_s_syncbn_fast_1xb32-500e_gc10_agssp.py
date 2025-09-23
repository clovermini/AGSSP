_base_ = './yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = '/home/data/Datasets/public/GC10-DET/'
class_name = ('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban','5_youban','6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhe')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette='random')   

work_dir = './work_dirs/own_pretrain/metal_ft_cos_frozen/yolov8_s_syncbn_fast_1xb32-500e_gc10/'

max_epochs = 500
train_batch_size_per_gpu = 32
train_num_workers = 4  

base_lr = _base_.base_lr / 4   
optim_wrapper = dict(optimizer=dict(lr=base_lr))  


load_from = '/home/data/XXX/projects/defect_detection/weights/own_pretrain/metal_ft_yolov8s_cos_frozen_epoch_10.pth'

model = dict(
    backbone=dict(frozen_stages=-1,  
                  #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
                  ), 
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
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='coco/bbox_mAP_50'),
    # The warmup_mim_iter parameter is critical.
    param_scheduler=dict(max_epochs=max_epochs),  # warmup_mim_iter=10
    #param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),  # warmup_mim_iter=10
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


