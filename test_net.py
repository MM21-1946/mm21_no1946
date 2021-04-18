import argparse
import os

import torch

from vmr.config import cfg
from vmr.data import make_data_loader
from vmr.engine.inference import inference
from vmr.modeling import build_model
from vmr.utils.checkpoint import TanCheckpointer
from vmr.utils.comm import synchronize, get_rank
from vmr.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="VMR")
    parser.add_argument(
        "--config-file",
        default="configs/2dtan_128x128_pool_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("vmr", cfg.OUTPUT_DIR, get_rank(), "log_test.txt")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    torch.backends.cudnn.enabled = cfg.MODEL.CUDNN
    model.to(cfg.MODEL.DEVICE)

    checkpointer = TanCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    k = 0
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        k += 1
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
            result_output_dir=cfg.OUTPUT_DIR + f'/eval_{dataset_name}_'
        )
        synchronize()

if __name__ == "__main__":
    main()
