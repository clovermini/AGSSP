_base_ = [
    '../_base_/datasets/industry_metal_bs256_swin_224_distill.py',
    '../_base_/default_runtime.py'
]

class_name = ('steel_pipe', 'BSD_cls', 'DAGM2007_Class10', 'neu_rail', 'aluminum', 'steel_rail', 'moderately_thick_plates', 'DAGM2007_Class9', 'wood', 'Marbled', 'Mesh', 'cold_rolled_strip_steel', 'severstal_steel', 
                'hot_rolled_strip_annealing_picking', 'Perforated', 'DAGM2007_Class7', 'Stratified', 'AITEX', 'Blotchy', 'BTech_02', 'bao_steel', 'Matted', 'KolektorSDD', 'BSData', 'medium_heavy_plate', 
                'aluminum_strip', 'DAGM2007_Class1', 'DAGM2007_Class6', 'leather', 'aluminum_ingot', 'neu_leather', 'DAGM2007_Class3', 'tianchi_aluminum', 'neu_aluminum', 'wide_thick_plate', 'gc10_steel_plate', 'Woven_127', 
                'rail_surface', 'neu_tile', 'Magnetic_tile', 'metal_plate', 'DAGM2007_Class4', 'Woven_068', 'grid', 'KolektorSDD2', 'Woven_104', 'road_crack', 'Woven_001', 'DAGM2007_Class8', 'neu_hot_rolled_strip', 
                'hot_rolled_strip_steel', 'neu_magnetic_tiles', 'Fibrous', 'neu_steel', 'DAGM2007_Class5', 'Woven_125', 'DAGM2007_Class2', 'ssgd_glasses', 'wukuang_medium_plate', 'nan_steel', 'tile')  # 61  'good', 'defect'
num_classes = len(class_name)

# dataset 2 x 256
train_dataloader = dict(batch_size=320, num_workers=8)

work_dir = './work_dirs/swin-base_1xb256_metal_distill_multi_frozen'

# swin base_224
# model settings
model = dict(
    type='ImageDistillClassifier',
    backbone=dict(
        type='SwinTransformer', arch='base', img_size=224, drop_path_rate=0.5, window_size=7, convert_weights=True, out_indices=(3,), frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='/XXX/defect_detection/weights/swin_base_patch4_window7_224.pth')
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original', loss_weight=0.0),  # 0.1
        cal_acc=False),
    distill_head=dict(
        type='MultiLayerDistillHead',
        loss=dict(
            type='L2Loss', loss_weight=2.0, normalize=True)),  
            #type='CosineSimilarityLoss', shift_factor=2.0, scale_factor=2.0)),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ]
)

find_unused_parameters = True 


optim_wrapper = dict(
    type='AmpOptimWrapper',   # # schedule settings fp16 amp setting
    loss_scale='dynamic',
    clip_grad=dict(max_norm=5.0),
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * 640 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)


default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=60, save_best='accuracy/top1'),
    logger=dict(type='LoggerHook', interval=10), 
    )

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])  # tensorboard 

randomness = dict(seed=2024, deterministic=True, diff_rank_seed=False)
