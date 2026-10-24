from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from modules.utils.common import cost_time, torch_float32
import modules.dataset as dataset
import modules.dataset.dataset_iter as dataset_iter
# import modules.dataset.collater as collater
from modules.utils.logging import logger
from copy import deepcopy
import random

@cost_time
def test_DataStore(conf):
    store_file = conf.get('store_file', 'data/train/noisy.flac.hdf5')
    data_store = dataset.DataStore(store_file)
    # data/train/noisy.flac.hdf5: 51.681 s, data/train/noisy.wav.hdf5: 3.838 s
    logger.info(store_file)
    for _ in tqdm(data_store):
        pass


@cost_time
def test_Dataset(conf):
    subset = conf.get('subset', 'train')
    kwargs = deepcopy(conf['data']['dataloader'])
    data_key = conf.get('data_key', 'noisy_wav')
    kwargs = OmegaConf.to_container(kwargs, resolve=True) # dict
    dataset_name = conf.pDataset or conf['data'][subset]['name']
    p_dataset = getattr(dataset, dataset_name)(conf, subset)
    # if subset in ['valid', 'test']:
    #     kwargs['batch_size'] = 1
    if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
        kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])()
    dataloader = DataLoader(p_dataset, **kwargs)
    batch = next(iter(dataloader))
    logger.debug(type(batch))
    logger.debug(batch.keys())
    logger.debug('{} {}'.format(batch[data_key].data.shape, 
                                batch[data_key].data.dtype))
    for _ in tqdm(dataloader): # travel through the data iterator
        pass


@cost_time
def test_NoiseDataset(conf):
    subset = conf.get('subset', 'train')
    kwargs = deepcopy(conf['data']['dataloader'])
    kwargs = OmegaConf.to_container(kwargs, resolve=True) # dict
    dataset_name = 'NoiseDataset'
    p_dataset = getattr(dataset, dataset_name)(conf, subset)
    dataloader = DataLoader(p_dataset, **kwargs)
    batch = next(iter(dataloader))
    logger.debug(type(batch))
    logger.debug(batch.shape)
    for _ in tqdm(dataloader): # travel through the data iterator
        pass


@cost_time
def test_read_label(conf):
    subset = conf['subset']
    label_scp = conf['data'][subset]['label_scp']
    n_split = conf['data']['n_split']
    label_data = dataset_iter.read_label(label_scp, n_split)
    logger.debug(f'label_data: {len(label_data)}, {str(random.choice(list(label_data.items())))} ...')


if __name__ == '__main__':
    conf = OmegaConf.create({
        'conf': 'conf/config.yaml',
        'cmd': 'test_Dataset',
        'pDataset': None,
        'subset': 'train'
    })
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.conf), conf)
    eval(conf.cmd)(conf)
