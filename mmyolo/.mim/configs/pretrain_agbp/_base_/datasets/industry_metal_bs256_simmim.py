# dataset settings
dataset_type = 'CustomDataset'
data_root = '/XXX/datasets/'
mean = [110.2955, 113.433, 109.914]
std = [32.028, 30.614, 30.27]

data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=mean,
    std=std,
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=192, crop_ratio_range=(0.67, 1.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(type='PackInputs')
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
        ann_file='collection_of_images/metal_images_for_pretrain.txt',
        data_prefix='',
        pipeline=train_pipeline))
