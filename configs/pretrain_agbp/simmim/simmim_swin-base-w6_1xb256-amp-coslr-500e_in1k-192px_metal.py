_base_ = [
    '../_base_/datasets/industry_metal_bs256_simmim.py',
    '../_base_/default_runtime.py',
]

# dataset 2 x 256
train_dataloader = dict(batch_size=320, num_workers=8)

work_dir = './work_dirs/simmim_swin-base-w6_1xb256-amp-coslr-500e_in1k-192px_metal'

load_from = '/XXX/defect_detection/weights/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth'

# model settings
model = dict(
    type='SimMIM',
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='base',
        img_size=192,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(
        type='SimMIMLinearDecoder', in_channels=128 * 2**3, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='PixelReconstructionLoss', criterion='L1', channel=3)))

# optimizer wrapper
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4 * 640 / 512,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.)
        }))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7 / 1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        milestones=[160],
        by_epoch=True,
        begin=10,
        end=200,
        convert_to_iter_based=True)
]

# runtime
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    # only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=20))

# randomness
randomness = dict(seed=2024, deterministic=True, diff_rank_seed=True)

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]) 

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
