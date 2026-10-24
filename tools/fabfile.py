import os
from pathlib import Path
from fabric import Connection, task, SerialGroup, ThreadingGroup
from fabric.exceptions import GroupException
from invoke.exceptions import UnexpectedExit, CommandTimedOut
from paramiko import SSHException
import time
from omegaconf import OmegaConf
from rich import print as rprint
from modules.utils.logging import logger
import random

def load_config(config_path='conf/fabric.yaml'):
    CONDA_ROOT = f'{Path.home()}/shared/miniconda3'
    default_config = OmegaConf.create({
        'nodes': ['192.168.88.201', '192.168.88.202',
                  '192.168.88.203', '192.168.88.204',
                  '192.168.88.205', '192.168.88.206'],
        'gpu': {'num': 1, 'mem': -1},
        'conda': {
            'activate': f'source {CONDA_ROOT}/bin/activate {CONDA_ROOT}/envs/', 
            'name': 'speech'},
        'task': {
            'command': 'python -m test.test_gpu',
            'poll_interval': 60,
            'pwd': os.getcwd()}
        })
    try:
        user_config = OmegaConf.load(config_path)
        return OmegaConf.merge(default_config, user_config)
    except Exception as e:
        logger.info(f'{str(e)}, using default_config')
        logger.info(f'Save default_config to {config_path}')
        OmegaConf.save(default_config, config_path)
        return default_config


def get_find_gpu_cmd(conf):
    return f'python -m tools.find_gpu num={conf.gpu.num} mem={conf.gpu.mem}'


def get_remote_cmd(cmd, conf):
    remote_cmd = f'{conf.conda.activate}{conf.conda.name}'
    remote_cmd += f' && cd {conf.task.pwd}'
    remote_cmd += f' && . path.sh && {cmd}'
    return remote_cmd


def get_node_avaliable_gpu(node, conf):
    try:
        with Connection(node, connect_timeout=15) as conn:
            remote_command = get_remote_cmd(get_find_gpu_cmd(conf), conf)
            result = conn.run(remote_command, hide=True, timeout=30)
            return result.stdout.strip()
    except Exception as e:
        logger.info(f'node {node} error: {str(e)}')
        return None


def get_avaliable_gpu(conf):
    group = ThreadingGroup(*conf.nodes, connect_timeout=15)
    remote_command = get_remote_cmd(get_find_gpu_cmd(conf), conf)
    logger.info(remote_command)
    result_dict = {}
    try:
        results = group.run(remote_command, hide=True, timeout=30)
        for node, result in results.items():
            if result.stdout.strip():
                result_dict[node.host] = result.stdout.strip()
    except GroupException as e:
        for node, result in e.result.items():
            if isinstance(result, (UnexpectedExit, CommandTimedOut, SSHException)):
                # logger.warning(f"{node.host}: {result.result.stderr.strip()}")
                continue
            if result.stdout.strip():
                result_dict[node.host] = result.stdout.strip()
    except Exception as e:
        logger.info(str(e))
    return result_dict


@task
def run_test(c):
    conf = load_config()
    logger.info(f'Work dir: {conf.task.pwd}')
    node = '192.168.88.205'
    result = get_node_avaliable_gpu(node, conf)
    logger.info(f'{node}: {result}')
    logger.info(get_avaliable_gpu(conf))


@task
def run_task(c, task_cmd=None, gpu_num=1, gpu_mem=-1, conda_name='speech', nodes=None, 
             poll_interval=60, data_cache=None):
    '''
    fab run-task --task-cmd "python xxx"
    fab run-task --n-gpu 2 --conda-env-name myenv --nodes 'node1,node2' --task-cmd "python xxx"
    '''
    conf = load_config()
    if nodes:
        conf.nodes = nodes.split(',')
    conf.conda.name = conda_name
    conf.task.poll_interval = poll_interval
    conf.gpu.num = gpu_num
    conf.gpu.mem = gpu_mem
    
    command = task_cmd if task_cmd else conf.task.command
    if gpu_num > 1 and ('gpu.num' not in command):
        command += f' gpu.num={gpu_num}'
    if data_cache:
        command += f' data.cache={data_cache}'
    logger.info(command)
    while True:
        node_dict = get_avaliable_gpu(conf)
        if len(node_dict) > 0:
            logger.info(node_dict)
            node, gpus = random.choice(list(node_dict.items()))
            command = f'CUDA_VISIBLE_DEVICES={gpus} {command}'
            try:
                with Connection(node) as conn:
                    remote_command = get_remote_cmd(command, conf)
                    logger.info(f'Run task on {node}\n{remote_command}')
                    conn.run(remote_command, pty=True)  # 实时输出日志
                    return
            except Exception as e:
                logger.error(str(e))
                return
        else:
            rprint(f'Querying GPUs per {conf.task.poll_interval} s ...')
            time.sleep(conf.task.poll_interval)