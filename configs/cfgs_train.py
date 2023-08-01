from easydict import EasyDict

cfgs = {
    # network
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'norm': 'instance',
    'use_dropout': True,
    'd_layers': 4,  # 3
    'direction': 'AtoB',

    # training
    'training': True,
    'total_epoch': 200,  
    'lr_decay_epoch': 100,
    'pool_size': 50,
    'gan_mode': 'lsgan',
    'lr': 0.0002,
    'lr_policy': 'linear',
    'beta1': 0.5,
    'lambda_identity': 0.1, # 0.5
    'lambda_A': 5,  # 10, 1
    'lambda_B': 5,  # 10, 1
    'tensorboard': '/cakes/tensorboard/prougatit',

    # dataset
    'dirA': '/keiki/ffhq/images1024x1024',
    'dirB': '/keiki/anime_face',
    'batchsize': 3,
    'worker': 5,
    'load_size': 256, #288,
    # 'crop_size': 256,


}

cfgs = EasyDict(cfgs)