# fp16 settings
# fp16 = dict(loss_scale=512.)

# model settings
model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    rbox_head=dict(
        type='S2ANetHead',
        num_classes=4,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        align_conv_type='AlignConv',#[AlignConv,DCN,GA_DCN]
        align_conv_size=3,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
        anchor_target_type='hbb_obb_rbox_overlap',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    odm_cfg=dict(
        anchor_target_type='obb_obb_rbox_overlap',
        anchor_inside_type='center',
        assigner=dict(
            type='MaxIoUAssignerRbox',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=3000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),
    max_per_img=3000)
dataset_type = 'CocoDotaOBBOnlyCARDataset'
data_root = '/content/gdrive/My Drive/Arirang/data/train/custom_coco_all_only_car/'
imgae_root = '/content/gdrive/My Drive/Arirang/data/train/coco_all/'
img_norm_cfg = dict(
    mean=[54.06, 53.295, 50.235], std=[36.72, 35.955, 33.915], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='RotatedRandomBrightness', Brightness_ratio=0.5),   
    dict(type='RotatedRandomGrayscale', Grayscale_ratio=0.3),
    dict(type='RotatedRandomColorTemperature', ColorTemperature=0.2),     
    #dict(type='RotatedRandomAffine', Affine_ratio=0.9), 
    dict(type='RandomRotate', rate=0.1, angles=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 235, 250, 265, 280, 295, 310, 325, 340], auto_bound=False),    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='RotatedRandomBrightness', flip_ratio=0.5),  
            dict(type='RotatedRandomAffine', Affine_ratio=0.95), 
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RotatedResize', img_scale=(1024, 1024), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
         ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=imgae_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=imgae_root + 'val2017/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file= '/content/gdrive/My Drive/Arirang/data/test/instances_test2017.json',
        img_prefix= '/content/gdrive/My Drive/Arirang/data/test/images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='Adam', lr=0.005, weight_decay=0.0001)
#dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[110])
checkpoint_config = dict(interval=8)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 6000
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/content/gdrive/My Drive/Arirang/models/s2anet_r50_fpn_1x_ms_rotate_olny_car_adam/'
load_from = '/content/gdrive/My Drive/Arirang/models/s2anet_r50_fpn_1x_ms_rotate_car_adam/latest.pth'#'/content/gdrive/My Drive/Arirang/models/s2anet_r50_fpn_1x_ms_rotate_adam/epoch_304.pth'#'/content/gdrive/My Drive/Arirang/s2anet_r50_fpn_1x_ms_rotate_epoch_12_20200815.pth'
resume_from = None#'/content/gdrive/My Drive/Arirang/models/s2anet_r50_fpn_1x_ms_rotate_car_adam/latest.pth'

workflow = [('train', 1)]
# r50
# map: 0.7897890609404231
# classaps:  [89.35435171 83.77469654 56.28060854 77.51105958 78.91526788 83.64872154
#  88.97497255 90.81995637 84.11129215 87.23208697 72.75258904 69.38579686
#  78.01568078 74.024542   69.8819689 ]
