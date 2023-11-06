import yaml

# with open('configs/data_specs.yaml') as fh:
#     data_specs = yaml.load(fh, Loader=yaml.FullLoader)

import random, torch, os
# Data loader parameters

data = dict(
        data_dir = "/data_hdd1/users/Talha/inat2021/", 
        # training time settings
        img_height= 224,
        img_width= 224,
        input_channels= 3,
        label_smoothing= 0.15,
        # only for training data
        Augment_data= True,
        # Augmentation Prbabilities should be same legth
        step_epoch=    [0, 10],
        geometric_aug = [0.3, 0.3],
        noise_aug =     [0.3, 0.3],

        Normalize_data = True,
        Shuffle_data = True,

        pin_memory=  True,
        num_workers= 1,
        prefetch_factor= 2,
        persistent_workers= True,
        # data_specs = data_specs,
    )

