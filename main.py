#%%
import os
# os.chdir(os.path.dirname(__file__))
os.chdir("/data_hdd1/users/khubaib/CWD30/")

from configs.config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];

if config['LOG_WANDB']:
    import wandb
    # from datetime import datetime
    # my_id = datetime.now().strftime("%Y%m%d%H%M")
    wandb.init(dir=config['log_directory'],
               project=config['project_name'], name=config['experiment_name'],
            #    resume='allow', id=my_id, # this one introduces werid behaviour in the app
               config_include_keys=config.keys(), config=config)
    # print(f'WANDB config ID : {my_id}')
import pprint
print(f'Printing Configuration File:\n{30*"="}\n')
pprint.pprint(config)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fmutils import fmutils as fmu

import imgviz, cv2, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from termcolor import cprint
from tqdm import tqdm
from itertools import cycle, chain
mpl.rcParams['figure.dpi'] = 300


from data.dataloader import GEN_DATA_LISTS, CWD26, FDALoader
from data.utils import collate

from core.backbones.mit import MixVisionTransformer
from core.backbones.mscan import MSCANet
from core.decoders.hrda_decoder import HRDAHead
from core.utils.metrics import ConfusionMatrix
from core.utils.lr_scheduler import LR_Scheduler
from core.utils.chkpt_manager import load_checkpoint, save_chkpt

from tools.training import Trainer
from tools.evaluation import Evaluator, eval_wrapper

import torch.nn.functional as F

from gray2color import gray2color

pallet = config['data']['data_specs']['pallet']
g2c = lambda x : gray2color(x, use_pallet='cityscape',
                            custom_pallet=np.asarray(pallet).reshape(1,-1,3)/255)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
print('Source Dataset\n')
src_data_lists = GEN_DATA_LISTS(config['data']['src_data_dir'], config['data']['sub_directories'])
src_train_paths, src_val_paths, _ = src_data_lists.get_splits()
classes = src_data_lists.get_classes()
src_data_lists.get_filecounts()
print()

print('Target Dataset\n')
trg_data_lists = GEN_DATA_LISTS(config['data']['trg_data_dir'], config['data']['sub_directories'])
trg_train_paths, _, trg_test_paths = trg_data_lists.get_splits()
trg_data_lists.get_filecounts()
# repeat same indices of images and labels 
indices = [i for i in random.choices(range(len(trg_train_paths[0])), k=len(src_train_paths[0]))]
trg_train_paths[0] = [trg_train_paths[0][i] for i in indices]
trg_train_paths[1] = [trg_train_paths[1][i] for i in indices]
print(f'Extended Traget domian data images: {len(trg_train_paths[0])}\n')

print('FDA Image\n') # using cycle to repeat the list makes it very slow
fda_data_list_orig = fmu.get_all_files(config['data']['fda_data_dir'])
fda_data_list = random.choices(fda_data_list_orig, k=len(src_train_paths[0]))
print(f'Number of FDA images Original: {len(fda_data_list_orig)}; Repeated: {len(fda_data_list)}')


train_data = CWD26(src_train_paths[0], src_train_paths[1], config['data']['img_height'], config['data']['img_width'],
                   config['data']['Augment_data'], config['data']['Normalize_data'])
train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                          # important for adaptive augmentation to work properly.
                          num_workers=config['data']['num_workers'], drop_last=True,
                          collate_fn=collate, pin_memory=config['data']['pin_memory'],
                          prefetch_factor=4, persistent_workers=True
                          )

val_data = CWD26(src_val_paths[0], src_val_paths[1], config['data']['img_height'], config['data']['img_width'],
                 False, config['data']['Normalize_data'])
val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['data']['num_workers'], drop_last=True,
                        collate_fn=collate, pin_memory=config['data']['pin_memory'],
                        prefetch_factor=4, persistent_workers=True
                        )


trg_train_data = CWD26(trg_train_paths[0], trg_train_paths[1], config['data']['img_height'], config['data']['img_width'],
                       config['data']['Augment_data'], config['data']['Normalize_data'])
trg_train_loader = DataLoader(trg_train_data, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['data']['num_workers'], drop_last=True,
                              collate_fn=collate, pin_memory=config['data']['pin_memory'],
                              prefetch_factor=4, persistent_workers=True
                              )

trg_test_data = CWD26(trg_test_paths[0], trg_test_paths[1], config['data']['img_height'], config['data']['img_width'],
                      False, config['data']['Normalize_data'])
trg_test_loader = DataLoader(trg_test_data, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['data']['num_workers'], drop_last=True,
                             collate_fn=collate, pin_memory=config['data']['pin_memory'],
                             prefetch_factor=4, persistent_workers=True
                             )

fda_data = FDALoader(fda_data_list, config['data']['img_height'], config['data']['img_width'],
                     config['data']['Normalize_data'])
fda_loader = DataLoader(fda_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['data']['num_workers'], drop_last=True,
                        collate_fn=None, pin_memory=config['data']['pin_memory'],
                        prefetch_factor=4, persistent_workers=True
                        )

if config['data']['sanity_check']:
    # DataLoader Sanity Checks
    batch = next(iter(train_loader))
    s=255
    img_ls = []
    [img_ls.append((batch['img'][i]*s).astype(np.uint8)) for i in range(config['batch_size'])]
    [img_ls.append(g2c(batch['lbl'][i])) for i in range(config['batch_size'])]
    plt.title('Sample Batch')
    plt.imshow(imgviz.tile(img_ls, shape=(4,config['batch_size']//2), border=(255,0,0)))
    plt.axis('off')
#%%
# encoder = MSCANet(**config['model']['encoder'])
# torch.backends.cudnn.benchmark = True

encoder = MixVisionTransformer(**config['model']['encoder'])
decoder = HRDAHead(config['model']['decoder'], config['model']['output_stride'],
                   config['model']['attention_classwise'])
if config['USE_EMA_UPDATES']:
    print('[INFO] Using EMA Updates...')
    ema_encoder = MixVisionTransformer(**config['model']['encoder'])
    ema_encoder.to(DEVICE)
else:
    ema_encoder = None

matched, unmatched = load_checkpoint(encoder, pretrained_path=f"{config['model']['checkpoint_path']}")
print(unmatched)

encoder.to(DEVICE)
decoder.to(DEVICE)

if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    # print(torch._dynamo.list_backends())
    # encoder = torch.compile(encoder, mode="max-autotune")
    # decoder = torch.compile(decoder, mode="max-autotune")
    if config['USE_EMA_UPDATES']:
        ema_encoder = nn.DataParallel(ema_encoder)
        # ema_encoder = torch.compile(ema_encoder, mode="max-autotune")

enc_optim = torch.optim.AdamW([{'params': encoder.parameters(),
                                'lr':config['enc_learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'])

dec_optim = torch.optim.AdamW([{'params': decoder.parameters(),
                                'lr':config['dec_learning_rate']}],
                                weight_decay=config['WEIGHT_DECAY'])

enc_scheduler = LR_Scheduler(config['lr_schedule'], config['enc_learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

dec_scheduler = LR_Scheduler(config['lr_schedule'], config['dec_learning_rate'], config['epochs'],
                         iters_per_epoch=len(train_loader), warmup_epochs=config['warmup_epochs'])

metric = ConfusionMatrix(config['data']['num_classes'])


trainer = Trainer(encoder, decoder, enc_optim, dec_optim,
                  metric, DEVICE, config['data'], ema_encoder)

evaluator = Evaluator(ema_encoder, decoder, metric, config['data'])
trg_evaluator = Evaluator(ema_encoder, decoder, metric, config['data'])

# Initializing plots
if config['LOG_WANDB']:
    wandb.watch(encoder, log='parameters', log_freq=100)
    wandb.watch(decoder, log='parameters', log_freq=100)
    wandb.log({"val_mIOU": 0, "mIOU": 0, "trg_mIOU": 0, 
               "loss": 10, "enc_learning_rate": 0,
               "dec_learning_rate": 0}, step=0)
#%%
start_epoch = 0
epoch, best_iou, curr_viou = 0, 0, 0
total_avg_viou, trg_total_avg_viou = [], []

for epoch in range(start_epoch, config['epochs']):
    if config['data']['apply_dacs']:
        pbar = tqdm(zip(train_loader, trg_train_loader, fda_loader), total=len(train_loader))
    else:
        pbar = tqdm(zip(train_loader, fda_loader), total=len(train_loader))
    encoder.train() # <-set mode important
    decoder.train()
    ta, tl = [], []
    
    for step, data_batch in enumerate(pbar):
        if config['data']['apply_dacs']:
            src_batch, trg_batch, fda_batch = data_batch
        else:
            trg_batch = src_batch
        enc_scheduler(enc_optim, step, epoch)
        dec_scheduler(dec_optim, step, epoch)
        
        loss_value = trainer.training_step(src_batch, trg_batch, fda_batch, epoch)
        iou = trainer.get_scores()
        trainer.reset_metric()
        
        tl.append(loss_value)
        ta.append(iou['iou_mean'])

        pbar.set_description(f'Epoch {epoch+1}/{config["epochs"]} - t_loss {loss_value:.4f} - mIOU {iou["iou_mean"]:.4f}')
        # break
    print(f'=> Average loss: {np.nanmean(tl):.4f}, Average IoU: {np.nanmean(ta):.4f}')
    g, n = src_batch['geo_augs'][0], src_batch['noise_augs'][0]
    

    if (epoch + 1) % 2 == 0: # eval every 2 epoch
        ema_encoder.eval()
        decoder.eval()
        curr_viou, avg_viou, total_avg_viou, tiled = eval_wrapper(evaluator, val_loader, total_avg_viou)
        trg_curr_viou, trg_avg_viou, trg_total_avg_viou, trg_tiled = eval_wrapper(trg_evaluator, trg_test_loader, trg_total_avg_viou)

        cprint(f'=> Averaged srcValidation IoU: {avg_viou:.4f}', 'magenta')
        cprint(f'=> Averaged trgValidation IoU: {trg_avg_viou:.4f}', 'red')

        if config['LOG_WANDB']:
            wandb.log({"val_mIOU": avg_viou, "trg_mIOU": trg_avg_viou}, step=epoch+1)
            wandb.log({'src_predictions': wandb.Image(tiled), 
                       'trg_predictions': wandb.Image(trg_tiled)}, step=epoch+1)

    if config['LOG_WANDB']:
        wandb.log({"loss": np.nanmean(tl), "mIOU": np.nanmean(ta),
                   "enc_learning_rate": enc_optim.param_groups[0]['lr'],
                   "dec_learning_rate": dec_optim.param_groups[0]['lr'],
                   'geo_augs': g, 'noise_augs': n}, step=epoch+1)
    
    if curr_viou > best_iou:
        best_iou = curr_viou
        save_chkpt(config, encoder, enc_optim, epoch, loss_value, best_iou, module='encoder')
        save_chkpt(config, decoder, dec_optim, epoch, loss_value, best_iou, module='decoder')
        if config['USE_EMA_UPDATES']:
            save_chkpt(config, ema_encoder, enc_optim, epoch, loss_value, best_iou, module='ema_encoder')
    # break

if config['LOG_WANDB']:
    wandb.run.finish()
#%%
