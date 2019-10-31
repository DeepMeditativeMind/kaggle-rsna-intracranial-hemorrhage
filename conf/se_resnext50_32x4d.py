from pathlib import Path
nm = Path(__file__).with_suffix("").name

#model = dict( name='se_resnext50_32x4d', pretrained=True, n_output=6,)
model = dict( name='se_resnext50_32x4d', pretrained=True, n_output=6,)
optim = dict( name='Adam', params=dict( lr=8e-5,),)
scheduler = dict( name='MultiStepLR', params=dict( milestones=[1,2], gamma=2/3,),)

n_fold = 5
epoch = 3
resume_from = None

batch_size = 32
num_workers = 8
imgsize = (512, 512)

from pathlib import Path
fn = Path(__file__)
workdir = f'./model/{nm}'
seed = 20
apex = True

normalize = None
loss = dict( name='BCEWithLogitsLoss', params=dict(),)
crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize[0], width=imgsize[1]))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5,))
vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)

window_policy = 2
loader_def = dict( num_workers=num_workers, pin_memory=False,)
data_def = dict( dataset_type='CustomDataset', imgsize=imgsize, dataset_policy='all', window_policy=window_policy,)

data = dict(
    train=dict(**data_def,
        annotations='./cache/train_folds.pkl',
        imgdir='./input/stage_1_train_images',
        n_grad_acc=1,
        loader=dict( shuffle=True, batch_size=batch_size, drop_last=True, **loader_def,),
        transforms=[crop, hflip, rotate, contrast, totensor],),
    valid = dict(**data_def,
        annotations='./cache/train_folds.pkl',
        imgdir='./input/stage_1_train_images',
        loader=dict( shuffle=False, batch_size=batch_size*2, drop_last=False, **loader_def,),
        transforms=[crop, hflip, rotate, contrast, totensor],),
    test = dict(**data_def,
        annotations='./cache/test.pkl',
        imgdir='./input/stage_1_test_images',
        loader=dict( shuffle=False, batch_size=batch_size*2, drop_last=False, **loader_def,),
        transforms=[crop, hflip, rotate, contrast, totensor],),
)
