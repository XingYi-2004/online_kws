import os
import sys
import torch
from pathlib import Path
from platform import node
from multiprocessing import cpu_count
import lightning as L
from datetime import datetime
from typing import Optional
from omegaconf import OmegaConf
from .common import get_ckpt, find_available_gpu, set_resource, is_kubernetes, merge_dicts
from modules.utils.logging import setup_root_logger, logger
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_float32_matmul_precision('high')
torch.set_num_threads(1)

def get_exp_name(conf):
    if 'exp_name' in conf['system']:
        return conf['system']['exp_name']
    model_conf = OmegaConf.to_container(conf['model'], resolve=True)
    if isinstance(model_conf, list):
        model_name = "_".join(item['name'] for item in model_conf)
    else:
        model_name = model_conf['name']
    return '{}_{}'.format(conf['system']['name'], model_name)

def set_cuda_visible_devices(conf):
    '''
    Try to find available GPU and set CUDA_VISIBLE_DEVICES    
    '''
    # Slurm will set CUDA_VISIBLE_DEVICES, or use all gpu in kubernetes
    if 'CUDA_VISIBLE_DEVICES' in os.environ or is_kubernetes():
        return
    if conf['cmd'] == 'train':  # train mode, find a gpu with no active processes
        default_conf = {'num': 1, 'mem': -1, 'interval': 60}
    else:  # otherwise, find a gpu with [mem]GB avaliable
        default_conf = {'num': 1, 'mem': 4, 'interval': -1}
    gpus = find_available_gpu(**merge_dicts(default_conf, conf.get('gpu', {})))
    if not gpus:
        logger.error('CUDA_VISIBLE_DEVICES not set, and find_available_gpu() failed.')
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def get_devices():
    devices = 'auto'
    if torch.cuda.is_available():
        vis = os.getenv('CUDA_VISIBLE_DEVICES')
        if vis is None or vis.strip() == '':
            devices = -1
        else:
            ids = [int(x) for x in vis.split(',') if x.strip()]
            devices = ids if len(ids) != torch.cuda.device_count() else -1
    return devices


def _resolve_path(value) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd().joinpath(path)
    return path.resolve()


def _cli_overrides(key: str) -> bool:
    prefix = f'{key}='
    return any(arg.startswith(prefix) for arg in sys.argv[1:])


def _find_hparams_path(conf, ckpt) -> Optional[Path]:
    candidates = []
    if ckpt is not None:
        ckpt_path = _resolve_path(ckpt)
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == 'checkpoints' else ckpt_path.parent
        candidates.append(run_dir.joinpath('hparams.yaml'))

    conf_path = conf.get('conf')
    if conf_path:
        conf_path = _resolve_path(conf_path)
        if conf_path.name == 'hparams.yaml':
            candidates.append(conf_path)

    root_dir = conf.get('root_dir')
    if root_dir:
        candidates.append(_resolve_path(root_dir).joinpath('hparams.yaml'))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    return None


def _set_default_root_dir(conf):
    if 'root_dir' in conf:
        return
    exp_name = get_exp_name(conf)
    version: str = conf['version'] if type(conf['version']) == str else str(conf['version'])
    conf['root_dir'] = str(Path(conf['system']['log_dir']).joinpath(exp_name, version))


def init_conf(conf):
    original_cmd = conf.get('cmd')
    original_root_dir = conf.get('root_dir')
    has_explicit_root_dir = 'root_dir' in conf

    # root_dir
    _set_default_root_dir(conf)

    # if ckpt not specified, try to find ckpt in root_dir
    ckpt = conf.get('ckpt')
    if ckpt is None:
        ckpt = get_ckpt(conf['root_dir'], conf['cmd'])
    hparams_path = _find_hparams_path(conf, ckpt) if ckpt is not None else None
    if hparams_path is not None:
        conf = OmegaConf.load(hparams_path)
        conf.merge_with_cli()  # merge again
        if original_cmd is not None:
            conf['cmd'] = original_cmd
        if has_explicit_root_dir:
            conf['root_dir'] = original_root_dir
        elif not _cli_overrides('root_dir'):
            conf['root_dir'] = str(hparams_path.parent)
        _set_default_root_dir(conf)

    Path(conf['root_dir']).mkdir(parents=True, exist_ok=True)
    setup_root_logger(conf['system']['debug'], Path(conf['root_dir']).joinpath('log.log'), conf['cmd'])
    logger.info('[Start] {} {} {}'.format('-'*10, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], '-'*10))
    if os.getenv('SLURM_JOB_ID'):
        logger.info('SLURM_JOB_ID: ' + os.getenv('SLURM_JOB_ID'))

    if ckpt is not None and hparams_path is None:
        logger.warning(
            'ckpt is set, but no hparams.yaml was found beside the checkpoint '
            'or under root_dir; using the current configuration.'
        )
    conf['ckpt'] = ckpt  # set ckpt
    logger.info('ckpt: {}'.format(conf['ckpt']))
    
    set_cuda_visible_devices(conf)
    logger.info('Hostname: {}, cpu_count: {}, CUDA_VISIBLE_DEVICES: [{}], HIP_VISIBLE_DEVICES: [{}], device_count: {}'.format(
        node(), cpu_count(), os.getenv('CUDA_VISIBLE_DEVICES'),  os.getenv('HIP_VISIBLE_DEVICES'),
        torch.cuda.device_count()))
    
    conf['trainer']['devices'] = get_devices()
    conf['trainer']['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'    
    set_resource()  # set resource limit
    L.seed_everything(conf['system'].get('seed', 0))  # seed
    # when validation is diable, disable ckpt_monitor and early_stop
    if conf['trainer'].get('limit_val_batches', 1.0) == 0:
        callbacks_conf = conf['trainer_callbacks']
        callbacks_conf['model_checkpoint']['monitor'] = None  # disable monitor
        callbacks_conf['model_checkpoint']['save_top_k'] = 1
        callbacks_conf['model_checkpoint']['save_last'] = False
        callbacks_conf['early_stop'] = None  # disable early_stop
    logger.debug(conf['trainer'])
    logger.debug(conf['trainer_callbacks'])
    return conf
