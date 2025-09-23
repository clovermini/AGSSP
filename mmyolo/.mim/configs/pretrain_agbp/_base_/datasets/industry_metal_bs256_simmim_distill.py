# dataset settings
dataset_type = 'CustomDistillDataset'
data_root = '/XXX/datasets/'
mean = [110.2955, 113.433, 109.914]
std = [32.028, 30.614, 30.27]

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=mean,
    std=std,
    to_rgb=True)

custom_imports = dict(imports=['mmseg.datasets'], allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageAnomalyMapData'),
    dict(  
        type='RandomResize',
        scale=(400, 400),
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='mmseg.RandomCrop', crop_size=192, cat_max_ratio=0.75),
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(type='PackAnomalyInputs')
]

train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='collection_of_images/pretrain_distill_1017/metal_images_for_pretrain_v3_pretrain_mini_pretrain.txt',
        data_prefix='',
        pipeline=train_pipeline))
