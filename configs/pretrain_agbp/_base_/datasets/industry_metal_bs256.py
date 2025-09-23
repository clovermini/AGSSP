# dataset settings
dataset_type = 'CustomDistillDataset'
data_root = '/XXX/datasets'  

class_name = ('steel_pipe', 'BSD_cls', 'DAGM2007_Class10', 'neu_rail', 'aluminum', 'steel_rail', 'moderately_thick_plates', 'DAGM2007_Class9', 'wood', 'Marbled', 'Mesh', 'cold_rolled_strip_steel', 'severstal_steel', 
                'hot_rolled_strip_annealing_picking', 'Perforated', 'DAGM2007_Class7', 'Stratified', 'AITEX', 'Blotchy', 'BTech_02', 'bao_steel', 'Matted', 'KolektorSDD', 'BSData', 'medium_heavy_plate', 
                'aluminum_strip', 'DAGM2007_Class1', 'DAGM2007_Class6', 'leather', 'aluminum_ingot', 'neu_leather', 'DAGM2007_Class3', 'tianchi_aluminum', 'neu_aluminum', 'wide_thick_plate', 'gc10_steel_plate', 'Woven_127', 
                'rail_surface', 'neu_tile', 'Magnetic_tile', 'metal_plate', 'DAGM2007_Class4', 'Woven_068', 'grid', 'KolektorSDD2', 'Woven_104', 'road_crack', 'Woven_001', 'DAGM2007_Class8', 'neu_hot_rolled_strip', 
                'hot_rolled_strip_steel', 'neu_magnetic_tiles', 'Fibrous', 'neu_steel', 'DAGM2007_Class5', 'Woven_125', 'DAGM2007_Class2', 'ssgd_glasses', 'wukuang_medium_plate', 'nan_steel', 'tile')  # 61  extra 'good', 'defect'
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette='random')  

mean = [110.2955, 113.433, 109.914]
std = [32.028, 30.614, 30.27]
batch_size = 256  #256
num_workers = 4 # 4

data_preprocessor = dict(  
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=mean,
    std=std,
    # convert image from BGR to RGB
    to_rgb=True,
)

custom_imports = dict(imports=['mmseg.datasets'], allow_failed_imports=False)

crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadImageAnomalyMapData'),
    dict(  
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=False),
    dict(type='mmseg.RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),   # mmseg: process anomaly map at the same time
    dict(type='mmseg.RandomFlip', prob=0.5),
    dict(type='mmseg.PhotoMetricDistortion'),
    dict(type='PackAnomalyInputs')
]

#train_pipeline = [
#    dict(type='LoadImageAnomalyMapData'),
#    dict(type='mmseg.Resize', scale=(224, 224), keep_ratio=False),
#    dict(type='PackAnomalyInputs')
#]

'''
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',   
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   
    dict(
        type='RandAugment',    
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',  
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]
'''

test_pipeline = [
    dict(type='LoadImageAnomalyMapData'),
    dict(type='mmseg.Resize', scale=(224, 224), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackAnomalyInputs')
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
'''

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='collection_of_images/pretrain_distill_1017/metal_images_for_pretrain_v3_pretrain_mini_train.txt',
        data_prefix='',
        with_label=True,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='collection_of_images/pretrain_distill_1017/metal_images_for_pretrain_v3_pretrain_mini_val.txt',
        data_prefix='',
        with_label=True,
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator