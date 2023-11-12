import timm
import torch
from torch import nn
from torchvision import models
from conf.model_config import model_dict
from timm.models import create_model


# CNN models
def  ResNet101(args):
    if args.pretrained:
        resnet101 = models.resnet101(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        resnet101 = models.resnet101(weights=None)
    resnet101.fc = nn.Linear(resnet101.fc.in_features, args.num_classes)
    
    return resnet101

def  EfficientNet_v2_m(args):
    if args.pretrained:
        efficientnet_v2_m = models.efficientnet_v2_m(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        efficientnet_v2_m = models.efficientnet_v2_m(weights=None)
    efficientnet_v2_m.classifier[1] = nn.Linear(efficientnet_v2_m.classifier[1].in_features, args.num_classes)

    return efficientnet_v2_m

def  MobileNet_v3_large(args):
    if args.pretrained:
        mobilenet_v3_large = models.mobilenet_v3_large(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
    mobilenet_v3_large.classifier[3] = nn.Linear(mobilenet_v3_large.classifier[3].in_features, args.num_classes)

    return mobilenet_v3_large

def  ResNext101_32x8d(args):
    if args.pretrained:
        resnext101_32x8d = models.resnext101_32x8d(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        resnext101_32x8d = models.resnext101_32x8d(weights=None)

    resnext101_32x8d.fc = nn.Linear(resnext101_32x8d.fc.in_features, args.num_classes)

    return resnext101_32x8d

# Transformer models
def  Swin_b(args):
    if args.pretrained:
        swin_b = models.swin_b(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        swin_b = models.swin_b(weights=None)
    swin_b.head = nn.Linear(swin_b.head.in_features, args.num_classes)

    return swin_b

def  ViT_l_32(args):
    if args.pretrained:
        vit_l_32 = models.vit_l_32(weights='DEFAULT')
        print("Model {args.model} loaded with pretrained weights")
    else:
        vit_l_32 = models.vit_l_32(weights=None)
    vit_l_32.heads[0] = nn.Linear(vit_l_32.heads[0].in_features, args.num_classes)

    return vit_l_32

def  MaxViT_t(args):

    if args.pretrained:
        maxvit_t = models.maxvit_t(weights='DEFAULT')
        print(f"Model {args.model} loaded with pretrained weights")
    else:
        maxvit_t = models.maxvit_t(weights=None)
    maxvit_t.classifier[5] = nn.Linear(maxvit_t.classifier[5].in_features, args.num_classes)

    return maxvit_t


# timm models
def timm_model(args):
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    if args.local_pretrained and args.pretrained:
        pt = False
        num_classes = args.pretrained_num_classes
    else:
        pt = args.pretrained
        num_classes = args.num_classes


    timm_model = create_model(
        args.model,
        pretrained=pt,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )

    # load the checkpoint if pretrained=True
    if args.local_pretrained and args.pretrained:
        checkpoint = torch.load(args.local_pretrained, map_location='cpu')
        timm_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Model {args.model} loaded with pretrained weights")

        # change the last layer
        if args.pretrained_num_classes != args.num_classes:
            timm_model.head = nn.Linear(timm_model.head.in_features, args.num_classes, bias=True)

    return timm_model

# map model name to function
model_to_func = {'resnet50': timm_model,
 'resnet18': timm_model,
 'resnet101': ResNet101,
 'efficientnet_b0': timm_model,
 'efficientnet_b3': timm_model,
 'mobilenet_v3_small': timm_model,
 'movilenet_v3_large': MobileNet_v3_large,
 'convnext_large': timm_model,
 'convnext_base': timm_model,
 'convnext_tiny': timm_model,
 'tiny_vit_21m_224': timm_model,
 'vit_base_patch16_224': timm_model,
 'vit_large_patch32_224': ViT_l_32,
 'cait_xxs36_224': timm_model,
 'cait_s24_224': timm_model,
 'swin_s3_tiny_224': timm_model,
 'swin_s3_base_224': Swin_b,
 'swin_large_patch4_window7_224': timm_model,
 'maxvit_small_tf_224': MaxViT_t,
 'maxvit_base_tf_224': timm_model,
 'coatnet_1_rw_224': timm_model,
 'coatnet_3_rw_224': timm_model,
 'efficientformer_l1': timm_model,
 'efficientformer_l3': timm_model,
 'efficientformer_l7': timm_model}


def get_model(args):
    # Get the corresponding name of model from model_dict
    model_name_f = model_dict[args.model]

    # Get the corresponding function from model_to_func
    func = model_to_func[model_name_f]

    # Get the model
    model = func(args=args)

    return model
