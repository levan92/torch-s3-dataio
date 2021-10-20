import sys
import os
from datetime import timedelta
import logging 

import torch.distributed as dist
import torch.multiprocessing as mp

def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout,
):
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    main_func(local_rank, *args)


if __name__ == '__main__':
    from torch_dataset import main

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    dist_url = f"tcp://127.0.0.1:{port}"
    world_size = 4
    machine_rank = 0

    print(f'Hello from master script')

    mp.spawn(
                _distributed_worker,
                nprocs=world_size,
                args=(
                    main,
                    world_size,
                    1,
                    machine_rank,
                    dist_url,
                    (),
                    timedelta(minutes=30),
                ),
                daemon=False,
            )
