from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from modules.utils.common import cost_time
import modules.dataset as dataset
import modules.dataset.data_store as data_store
# import modules.dataset.collater as collater
from modules.utils.logging import logger
from copy import deepcopy
from pathlib import Path
import pytorch_lightning as pl
import psutil
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
torch.set_float32_matmul_precision('high')

@cost_time
def test_DataStore(conf):
    key = conf.get('key')
    store_class = conf.get('store_class', 'WavStore')
    if key is None:
        # data/train/noisy.flac.hdf5: 51.681 s, data/train/noisy.wav.hdf5: 3.838 s
        store_list = conf.get('store_file', 'data/train/noisy.flac.hdf5').split(',')
    else:
        store_list = OmegaConf.select(conf, key)
    logger.info(store_list)
    for store in store_list:
        store = Path(conf.data.data_dir).joinpath(store)
        logger.info(store)
        iter_obj = getattr(data_store, store_class)(store)
        for _ in tqdm(iter_obj):
            pass

class CustomTQDMProgressBar(TQDMProgressBar):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        process = psutil.Process()
        memory_info = process.memory_info()
        rss = memory_info.rss / (1024 ** 3)
        self.train_progress_bar.set_postfix({'mem': f'{rss:.2f} GB'})
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        
class LitDataLoader(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.dataloader_conf = conf['data']['dataloader']
        self.model = torch.nn.Linear(2, 1)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        if self.conf['system']['debug']:
            print(batch)
        return None

    def train_dataloader(self):
        subset = 'train'
        name = self.conf['data'][subset]['name']
        p_data = getattr(dataset, name)(self.conf, subset)
        logger.info(f'[{subset}] {type(p_data).__name__}')
        kwargs = OmegaConf.to_container(self.dataloader_conf, resolve=True) # dict
        if 'collate_fn' in kwargs and kwargs['collate_fn'] is not None:
            kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])(self.conf)
        return DataLoader(p_data, **kwargs)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

@cost_time
def test_LitDataLoader(conf):
    model = LitDataLoader(conf)
    conf['trainer']['devices'] = -1 if torch.cuda.is_available() else 'auto'
    conf['trainer']['accelerator'] = 'gpu' if torch.cuda.is_available() else 'cpu'    
    trainer = pl.Trainer(**conf['trainer'], callbacks=[CustomTQDMProgressBar()])
    trainer.fit(model)

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
        kwargs['collate_fn'] =  getattr(dataset, kwargs['collate_fn'])(conf)
    dataloader = DataLoader(p_dataset, **kwargs)
    batch = next(iter(dataloader))
    logger.debug(type(batch))
    logger.debug(batch.keys())
    logger.debug('{} {}'.format(batch[data_key].data.shape, 
                                batch[data_key].data.dtype))
    
    # for batch in tqdm(dataloader): # travel through the data iterator
    #     del batch
    process = psutil.Process()
    with tqdm(dataloader) as pbar:
        for batch in pbar:
            del batch
            mem_usage = process.memory_info().rss / 1024**3
            pbar.set_postfix({'mem': f'{mem_usage:.2f} GB'})
            

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
