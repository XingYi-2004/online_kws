#!/usr/bin/env python3

import argparse
import logging
logging.basicConfig(level = logging.INFO)
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import k2
except ImportError:
    pass
import torch
import torch.nn as nn
from asr_datamodule import SpeechCommandDataModule
from model_ann import TdnnLstm
from spikingjelly.activation_based import functional

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.decode import get_lattice, nbest_decoding, one_best_decoding
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    get_texts,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=19,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=5,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )
    parser.add_argument(
        "--export",
        type=str2bool,
        default=False,
        help="""When enabled, the averaged model is saved to
        tdnn/exp/pretrained.pt. Note: only model.state_dict() is saved.
        pretrained.pt contains a dict {"model": model.state_dict()},
        which can be loaded by `icefall.checkpoint.load_checkpoint()`.
        """,
    )
    parser.add_argument(
        "--num-class",
        type=int,
        default=29,
        help="Number of classes to classify",
    )
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "exp_dir": Path("tdnn_lstm_6ann_768_new2/exp/"),
            "lang_dir": Path("data/lang_phone"),
            "lm_dir": Path("data/lm"),
            # parameters for tdnn_lstm_ctc
            "subsampling_factor": 3,
            "feature_dim": 80,
            # parameters for decoding
            # "search_beam": 10,
            # "output_beam": 4,
            # "min_active_states": 16,
            # "max_active_states": 1000,
            "search_beam": 20,
            "output_beam": 7,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
        }
    )
    return params


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    device,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
    """
    results = []

    correct_num = 0
    total_num = 0

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        # ground truth label
        batch_label_list = []
        batch_size = len(batch["supervisions"]["text"])
        total_num += batch_size
        for i in range(batch_size):
            batch_label_list.append(int(batch["supervisions"]["cut"][i].supervisions[0].speaker))
        batch_label_list = torch.tensor(batch_label_list).to(device)

        feature = batch["inputs"]
        # at entry, feature is [N, T, C]
        feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
        assert feature.ndim == 3
        feature = feature.to(device)

        with torch.no_grad():
            out_fr = model(feature)

        correct_num += (out_fr.argmax(1) == batch_label_list).float().sum().item()

    acc = correct_num / total_num
    return acc


@torch.no_grad()
def main():
    parser = get_parser()
    SpeechCommandDataModule.add_arguments(parser)
    args = parser.parse_args()

    params = get_params()
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-decode")
    logging.info("Decoding started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    model = TdnnLstm(
        num_features=params.feature_dim,
        num_classes=params.num_class,  # +1 for the blank symbol
        subsampling_factor=params.subsampling_factor,
    )
    
    if params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                fname = f"{params.exp_dir}/epoch-{i}.pt"
                # 跳过损坏的检查点文件
                try:
                    torch.load(fname, map_location="cpu", weights_only=False)
                    filenames.append(fname)
                except Exception as e:
                    logging.warning(f"跳过损坏的检查点 {fname}: {e}")
        logging.info(f"averaging {filenames}")
        if len(filenames) == 0:
            raise RuntimeError("没有可用的检查点文件，全部损坏或不存在")
        model.load_state_dict(average_checkpoints(filenames))

    if params.export:
        logging.info(f"Export averaged model to {params.exp_dir}/pretrained.pt")
        torch.save({"model": model.state_dict()}, f"{params.exp_dir}/pretrained.pt")
    

    load_checkpoint(f"{params.exp_dir}/pretrained.pt", model)

    model.to(device)
    model.eval()

    # we need cut ids to display recognition results.
    args.return_cuts = True
    command = SpeechCommandDataModule(args)
    test_cuts = command.test_cuts()
    test_dl = command.test_dataloaders(test_cuts)
    # CAUTION: `test_sets` is for displaying only.
    # If you want to skip test-clean, you have to skip
    # it inside the for loop. That is, use
    #
    #   if test_set == 'test-clean': continue
    #
    test_sets = ["test"]
    test_dls = [test_dl]

    for test_set, test_dl in zip(test_sets, test_dls):
        acc = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            device=device
        )

        logging.info(f"Test set accuracy: {acc}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
