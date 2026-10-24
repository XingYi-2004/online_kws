from omegaconf import OmegaConf
from modules.utils.common import find_available_gpu

if __name__ == '__main__':
    # default conf: find a gpu with no active processes
    conf = OmegaConf.create({"num": 1, "mem": -1, "interval": -1})
    conf.merge_with_cli()
    print(find_available_gpu(**conf))
