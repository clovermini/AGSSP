_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/industry_metal_bs256.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

class_name = ('steel_pipe', 'BSD_cls', 'DAGM2007_Class10', 'neu_rail', 'aluminum', 'steel_rail', 'moderately_thick_plates', 'DAGM2007_Class9', 'wood', 'Marbled', 'Mesh', 'cold_rolled_strip_steel', 'severstal_steel', 
                'hot_rolled_strip_annealing_picking', 'Perforated', 'DAGM2007_Class7', 'Stratified', 'AITEX', 'Blotchy', 'BTech_02', 'bao_steel', 'Matted', 'KolektorSDD', 'BSData', 'medium_heavy_plate', 
                'aluminum_strip', 'DAGM2007_Class1', 'DAGM2007_Class6', 'leather', 'aluminum_ingot', 'neu_leather', 'DAGM2007_Class3', 'tianchi_aluminum', 'neu_aluminum', 'wide_thick_plate', 'gc10_steel_plate', 'Woven_127', 
                'rail_surface', 'neu_tile', 'Magnetic_tile', 'metal_plate', 'DAGM2007_Class4', 'Woven_068', 'grid', 'KolektorSDD2', 'Woven_104', 'road_crack', 'Woven_001', 'DAGM2007_Class8', 'neu_hot_rolled_strip', 
                'hot_rolled_strip_steel', 'neu_magnetic_tiles', 'Fibrous', 'neu_steel', 'DAGM2007_Class5', 'Woven_125', 'DAGM2007_Class2', 'ssgd_glasses', 'wukuang_medium_plate', 'nan_steel', 'tile')  # 61  'good', 'defect'
num_classes = len(class_name)

work_dir = './work_dirs/yolov8s_1xb256_metal'

custom_imports = dict(imports=['mmyolo.models'], allow_failed_imports=False)


# dataset 2 x 224
train_dataloader = dict(batch_size=1024, num_workers=8)

# schedule settings fp16 amp setting
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

load_from = '/XXX/defect_detection/weights/yolov8s-resnet_cls.pth'

last_stage_out_channels = 1024
deepen_factor = 0.33
widen_factor = 0.5
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmyolo.YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    head=dict(num_classes=num_classes, in_channels=512))

default_hooks = dict(
    checkpoint=dict(interval=50, max_keep_ckpts=50, save_best='accuracy/top1'),
    logger=dict(type='LoggerHook', interval=10), 
    )

# pip install tensorboard
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])  # tensorboard 

randomness = dict(seed=2024, deterministic=True, diff_rank_seed=False)
