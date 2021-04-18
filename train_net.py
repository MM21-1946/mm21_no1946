import argparse
import os

import torch
from torch import optim
from torch import multiprocessing
#multiprocessing.set_sharing_strategy('file_system') # May Causing dead lock when training finished

from vmr.config import cfg
from vmr.data import make_data_loader
from vmr.data.datasets.evaluation import evaluate_new
from vmr.engine.inference import inference, inference_new
from vmr.engine.trainer import do_train
from vmr.modeling import build_model
from vmr.utils.checkpoint import TanCheckpointer
from vmr.utils.comm import synchronize, get_rank
from vmr.utils.imports import import_file
from vmr.utils.logger import setup_logger
from vmr.utils.miscellaneous import mkdir, save_config


def train(cfg, local_rank, distributed):
    
    data_loader_train = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )
    
    model = build_model(cfg, data_loader_train.dataset)
    torch.backends.cudnn.enabled = cfg.MODEL.CUDNN
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    
    optimizer = getattr(optim, cfg.SOLVER.OPTIMIZER)(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    if cfg.SOLVER.SCHEDULER == '':
        scheduler = None
    else:
        scheduler = getattr(optim.lr_scheduler, cfg.SOLVER.SCHEDULER)(optimizer, milestones=cfg.SOLVER.MILESTONES)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = TanCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if cfg.MODEL.WEIGHT == "":
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    else:
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    
    arguments = {"epoch": 1, "best_epoch": 1, "best_r1iou": 0.0}
    arguments.update(extra_checkpoint_data)
    
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        data_loaders_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    
    best_epoch, _ = do_train(
        cfg,
        model,
        data_loader_train,
        data_loaders_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
    )

    return model

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    """
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
            result_output_dir=cfg.OUTPUT_DIR
        )
        synchronize()
    """
    dataset_list = []
    predictions_list = []
    for k, (dataset_name, data_loader_val) in enumerate(zip(dataset_names, data_loaders_val)):
        if k<2:
            dataset, predictions = inference_new(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            dataset_list.append(dataset)
            predictions_list.append(predictions)
        else:
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
                result_output_dir=cfg.OUTPUT_DIR + f'/val{k+1}_'
            )
        synchronize()
    evaluate_new(dataset_list, predictions_list, cfg.TEST.NMS_THRESH, result_output_dir=cfg.OUTPUT_DIR + f'/val1o2rp_')

def main():
    parser = argparse.ArgumentParser(description="VMR")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        help="run gpu test program on id",
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("vmr", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
