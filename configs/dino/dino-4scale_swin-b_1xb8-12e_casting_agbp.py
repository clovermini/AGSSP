_base_ = [
    './_base_/datasets/pipe_detection.py', './_base_/default_runtime.py'
]

data_root = '/home/data/Datasets/public/casting_billet/'
class_name = ('Sc', 'Ws', 'Co', 'WSM', 'Ss', 'Lc')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette='random') 
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

work_dir = './work_dirs/own_pretrain/metal_simmim_distill_multi_frozen/dino-4scale_swin-b_1xb8-12e_casting_billet'  # head_finetune with coco

max_epochs = 500
train_batch_size_per_gpu = 6
train_num_workers = 4  


load_from = '/home/data/XXX/projects/defect_detection/weights/own_pretrain/metal_simmim_distill_multi_frozen_epoch_200.pth'

find_unused_parameters=True
channels = [256, 512, 1024]
model = dict(
    type='DINO',
    num_queries=100,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=4,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[101.57, 101.57, 101.57],  # noqa  1 default
        std=[61.2, 61.2, 61.2],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        #_delete_=True, # Delete the backbone field in _base_
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
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=2,
        #init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
        ),   # , prefix='backbone.'
    neck=dict(
        type='ChannelMapper',
        in_channels=channels,
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=-0.5,  # -0.5 for DeformDETR  0.0 for ori
        temperature=10000),  # 10000 for DeformDETR  20 for ori
    bbox_head=dict(
        type='DINOHead',
        num_classes=num_classes,  # noqa
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),  # 2.0 in DeformDETR  1.0 for origin
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=300)),  # TODO: half num_dn_queries  100 for ori
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 640), (480, 640), (768, 640)],
                    #scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    #        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    #        (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 640), (500, 640), (600, 640)],
                    #scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 640), (480, 640), (768, 640)],  # 960
                    #scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    #        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    #        (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR  0.0001 for ori
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        })
    #paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), 
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_mini_500.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))

#_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'annotations/val.json', classwise=True)
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=1, save_best='coco/bbox_mAP_50',type='CheckpointHook'),
    # The warmup_mim_iter parameter is critical.
    # param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5), 
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=100,
        min_delta=0.005))

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[400],
        gamma=0.1)
]

randomness = dict(seed=2024, deterministic=False, diff_rank_seed=False)

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])  # tensorboard 




