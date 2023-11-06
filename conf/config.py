from conf import config_data

config = dict(
                gpus_to_use = '3',
                DPI = 300,
                LOG_WANDB= True,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = True,
                alpha = 0.999,

                project_name= 'CWDv7',
                experiment_name= 'Exp7_EMAC_aug0.7_lossgamma5',

                log_directory= "/home/user01/data/talha/CWD/logs/",
                checkpoint_path= "/home/user01/data/talha/CWD/chkpts/",

                # training settings
                batch_size= 2,
                WEIGHT_DECAY= 0.00005,
                # AUX_LOSS_Weights= 0.4,

                # Regularization SD 0.5 LS 1e-2
                stochastic_drop_path= 5e-1,
                SD_mode= 'batch',
                layer_scaling_val= 1e-5,

                # learning rate
                enc_learning_rate= 6e-05,
                dec_learning_rate= 6e-04,
                lr_schedule= 'cos',
                epochs= 100 ,
                start_epoch= 0,
                warmup_epochs= 1,
                # one of 'batch_norm' or 'sync_bn' or 'layer_norm'
                norm_typ= 'sync_bn',
                BN_MOM= 0.1,
                SyncBN_MOM= 0.1,
                # Loss Hyperparameters
                gamma = 5,
                lambda_d = 0.1,
                # model = config_model.hrda,
                data = config_data.data,
                )