import datetime
import logging
import os
import time

import torch
import torch.distributed as dist

from vmr.data import make_data_loader
from vmr.utils.comm import is_main_process, get_world_size, synchronize
from vmr.utils.metric_logger import MetricLogger
from vmr.engine.inference import inference

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss

def do_train(
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
):
    logger = logging.getLogger("vmr.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH
    max_iteration = len(data_loader_train)
    print_every_iter = max_iteration // 5

    model.train()
    start_training_time = time.time()
    end = time.time()

    for epoch in range(arguments["epoch"], max_epoch + 1):
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        for iteration, (batches, targets, _) in enumerate(data_loader_train):
            iteration += 1
            data_time = time.time() - end

            batches = batches.to(device)
            targets = targets.to(device)
            
            def closure():
                optimizer.zero_grad()
                loss_dict = model(batches, targets)
                if iteration % print_every_iter == 0 or iteration == max_iteration:
                    loss_dict_detached = {
                        k:reduce_loss(v.detach()) for k, v in loss_dict.items()
                    }
                    meters.update(**loss_dict_detached)
                loss_dict['total_loss'].backward()
                return loss_dict['total_loss']

            optimizer.step(closure)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % print_every_iter == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                
        if scheduler:
            scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)

        if data_loaders_val is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            for k, (testset_name, data_loader_val) in enumerate(zip(cfg.DATASETS.TEST, data_loaders_val)):
                r1_iou07_result = inference(
                    model,
                    data_loader_val,
                    dataset_name=testset_name,
                    nms_thresh=cfg.TEST.NMS_THRESH,
                    device=cfg.MODEL.DEVICE,
                    result_output_dir=cfg.OUTPUT_DIR + f'/{k+1}_{epoch}e_'
                )
                if is_main_process() and k == 0:
                    r1iou_1st_test = r1_iou07_result # use iou of the first test set
            synchronize()
            if is_main_process() and r1iou_1st_test > arguments["best_r1iou"]:
                arguments["best_epoch"] = epoch
                arguments["best_r1iou"] = r1iou_1st_test
                checkpointer.save(f"model_{epoch}e", **arguments)
            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )

    checkpointer.load("model_{}e".format(arguments["best_epoch"]), **arguments)

    return arguments["best_epoch"], arguments["best_r1iou"]