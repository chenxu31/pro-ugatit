from easydict import EasyDict

cfgs = {
    # model
    'inc': 1,
    'outc': 1,
    'ngf': 32,
    'ndf': 32,
    'use_dropout': False,
    'n_blocks': 2,
    'd_layers': 3,
    'training': True,
    # dataset
    'anime': False,  # to ensure dataset alignment.
    'worker': 5,
    # training
    'total_epoch': 100,  # 100->200
    'tensorboard': '/cakes/tensorboard/prougatit',
    'resume': '',   # resume training.
    'start_epoch': 0,  # if resume, please set start epoch.
    'saved_dir': '',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 5e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 1,
    'outc': 1,
    'ngf': 32,
    'ndf': 32,
    'use_dropout': False,
    'n_blocks': 2,
    'd_layers': 3,
    'training': False,
})