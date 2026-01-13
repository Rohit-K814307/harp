import yaml
from box import Box

def load_config(path):
    with open(path, 'r') as f:
        args = Box(yaml.safe_load(f))

        if hasattr(args, "lr"):
            args.lr = float(args.lr)

    return args