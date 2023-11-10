import timm
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters

def get_model(model_name, args):

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]


    ptcfg = timm.models._registry.get_pretrained_cfg(args.model)
    if args.local_pretrained and args.pretrained:
        ptcfg.url = None
        ptcfg.file = args.local_pretrained
        ptcfg.num_classes = args.pretrained_num_classes


    model = create_model(
        args.model,
        pretrained=args.pretrained,
        pretrained_cfg=ptcfg,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        # checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )