import timm
from timm import utils

model_name = 'resnet101.a1_in1k'
ptcfg = timm.models._registry.get_pretrained_cfg(model_name)
ptcfg.url = None
pretrained_weights = '../output/train/20231108-211157-vit_base_patch16_224_augreg2_in21k_ft_in1k-224/checkpoint-12.pth.tar'
ptcfg.file = pretrained_weights
ptcfg.num_classes = 4271
# print(ptcfg)


# load timm resnet model with a checkpoint
model = timm.create_model(
    model_name,
    pretrained=True,
    # pretrained_cfg=ptcfg,
    in_chans=3,
    num_classes=4271,
)

# from timm.models import load_checkpoint

# load_checkpoint(model, pretrained_weights)