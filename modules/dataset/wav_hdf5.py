import h5py
from modules.dataset.common import read_csv
from modules.utils.common import get_pool_executor
from modules.utils.logging import logger
from tools.prepare_data import get_wav
from pathlib import Path
import numpy as np
from tqdm import tqdm

pool = get_pool_executor()

class WavHDF5:
    def __init__(self, **kwargs):
        self.reader_list = []
        self.hdf5_file_list = []
        self.base_dir = kwargs.get('base_dir')

    def get_data(self, wav_path):
        # wav_path: wav.hdf5:inner_idx, or wav.hdf5:inner_idx|wav.hdf5:inner_idx
        if '|' in wav_path:
            return np.concatenate([self.get_data(x) for x in wav_path.split('|')])
        if self.base_dir:
            wav_path = self.base_dir + wav_path
        hdf5_file, inner_idx = wav_path.split(':')
        reader = self.__get_reader(hdf5_file)
        return reader['wav_data'][int(inner_idx)][:]

    def __get_reader(self, hdf5_file):
        # get the hdf5 reader
        if hdf5_file in self.hdf5_file_list:
            idx = self.hdf5_file_list.index(hdf5_file)
            reader = self.reader_list[idx]
        else:
            self.hdf5_file_list.append(hdf5_file)
            reader = h5py.File(hdf5_file, "r")
            self.reader_list.append(reader)
        return reader

    def __init_writer(self, writer, n, sample_rate=None, format=None):
        writer.create_dataset('wav_data', (n,), dtype=h5py.vlen_dtype(np.int16), 
                              compression='gzip')
        writer.create_dataset('wav_ids', (n,), dtype=h5py.string_dtype())
        if sample_rate:
            writer.attrs['sample_rate'] = sample_rate
        if format:
            writer.attrs['format'] = format
        return writer

    def __init_writers(self, hdf5_file, n_wav, sample_rate=16000, format='wav', n_split=1):
        each_split = n_wav // n_split
        remainder = n_wav % n_split
        self.n_list = [each_split + 1] * remainder + [each_split] * (n_split - remainder)
        self.n_cumsum = np.cumsum(self.n_list)
        logger.info(f'n_list: {self.n_list}, n_cumsum: {self.n_cumsum}')
        self.writer_list = []
        if n_split == 1:
            writer = h5py.File(hdf5_file, 'w')
            self.hdf5_file_list.append(hdf5_file)
            self.writer_list.append(self.__init_writer(writer, n_wav, sample_rate, format))
        else:
            hdf5_file_dir = hdf5_file.parent / f'{hdf5_file.stem}_hdf5.split{n_split}'
            hdf5_file_dir.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(self.n_list):
                hdf5_file_i = hdf5_file_dir / f'{i}{hdf5_file.suffix}'
                self.hdf5_file_list.append(hdf5_file_i)
                writer = h5py.File(hdf5_file_i, 'w')
                self.writer_list.append(self.__init_writer(writer, n, sample_rate, format))

    def __write_data(self, wav_data, wav_id, idx):
        outer_idx = np.searchsorted(self.n_cumsum, idx, side='right')
        inner_idx = idx - self.n_cumsum[outer_idx] + self.n_list[outer_idx]
        # outer_idx, inner_idx = idx // self.n_list[0], idx % self.n_list[0] # bug !
        # logger.info(f'{outer_idx}/ {inner_idx}')
        self.writer_list[outer_idx]['wav_data'][inner_idx] = wav_data
        self.writer_list[outer_idx]["wav_ids"][inner_idx] = str(wav_id)
        return outer_idx, inner_idx

    def __close_writers(self):
        for writer in self.writer_list:
            writer.close()

    def wav_to_hdf5(self, csv_file, hdf5_file, n_split=1):
        # write wav_path in csv_file to hdf5 with n_split
        hdf5_file = Path(hdf5_file)
        df = read_csv(csv_file)
        self.__init_writers(hdf5_file, len(df), n_split=n_split)
        future_tasks = []
        wav_path_dict = {}
        for idx, (wav_id, row) in enumerate(df.iterrows()):
            future_tasks.append(pool.submit(get_wav, row["wav_path"], idx, wav_id))
        for f in tqdm(future_tasks):
            wav_id, wav_data, idx = f.result()
            outer_idx, inner_idx = self.__write_data(wav_data, wav_id, idx)
            wav_path_dict[wav_id] = f'{self.hdf5_file_list[outer_idx]}:{inner_idx}'
        self.__close_writers()
        df['wav_path'] = df.index.map(wav_path_dict)
        csv_file = Path(csv_file)
        out_csv = csv_file.with_stem(f'{csv_file.stem}_hdf5')
        logger.info(f'out_csv: {out_csv}')
        df.to_csv(out_csv)

    def convert_hdf5(self, csv_file, hdf5_file, src_file, n_split=1, group=None):
        # convert the old version (src_file) to the new version (hdf5_file)
        hdf5_file = Path(hdf5_file)
        df = read_csv(csv_file)
        reader = h5py.File(src_file, 'r')
        if group:
            reader = reader[group]
        self.__init_writers(hdf5_file, len(df), n_split=n_split)
        wav_path_dict = {}
        for idx, (wav_id, _) in tqdm(enumerate(df.iterrows()), total=len(df)):
            wav_data = reader[wav_id][:]
            outer_idx, inner_idx = self.__write_data(wav_data, wav_id, idx)
            wav_path_dict[wav_id] = f'{self.hdf5_file_list[outer_idx]}:{inner_idx}'
        self.__close_writers()
        df['wav_path'] = df.index.map(wav_path_dict)
        csv_file = Path(csv_file)
        out_csv = csv_file.with_stem(f'{csv_file.stem}_hdf5')
        logger.info(f'out_csv: {out_csv}')
        df.to_csv(out_csv)