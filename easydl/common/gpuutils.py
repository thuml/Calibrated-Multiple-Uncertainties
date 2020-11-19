import torch
import sys

dist = sys.argv and any(['--local_rank' in x for x in sys.argv])


def get_available_GPUs(N, max_utilization=.5, max_memory_usage=.5):
    '''
    get `N` available GPU ids with *utilization* less than `max_utilization` and *memory usage* less than max_memory_usage
    Arguments:
        N (int): How many GPUs you want to select
        max_utilization (float): GPU with utilization higher than `max_utilization` is considered as not available.
        max_memory_usage (float): GPU with memory usage higher than `max_memory_usage` is considered as not available.

    Returns:
        list containing IDs of available GPUs
    '''
    from subprocess import Popen, PIPE
    cmd = ["nvidia-smi",
           "--query-gpu=index,utilization.gpu,memory.total,memory.used",
           "--format=csv,noheader,nounits"]
    p = Popen(cmd, stdout=PIPE)
    output = p.stdout.read().decode('UTF-8')
    gpus = [[int(x) for x in line.split(',')] for line in output.splitlines()]
    gpu_ids = []
    for (index, utilization, total, used) in gpus:
        if utilization / 100.0 < max_utilization:
            if used * 1.0 / total < max_memory_usage:
                gpu_ids.append(index)
    if len(gpu_ids) < N:
        raise Exception("Only %s GPU(s) available but %s GPU(s) are required!" % (len(gpu_ids), N))
    available = gpu_ids[:N]
    return list(available)


def select_GPUs(N_per_process, max_utilization=.5, max_memory_usage=.5):
    '''
    select `N_per_process` GPUs.
    If distributed training is enabled, GPUs will be assigned properly among different processes.
    Arguments:
        N_per_process (int): How many GPUs you want to select for each process
        max_utilization (float): GPU with utilization higher than `max_utilization` is considered as not available.
        max_memory_usage (float): GPU with memory usage higher than `max_memory_usage` is considered as not available.

    Returns:
        list containing IDs of selected GPUs
    '''
    if not dist:
        return get_available_GPUs(N_per_process, max_utilization, max_memory_usage)
    try:
        rank = torch.distributed.get_rank()
    except Exception as e:
        print('please call torch.distributed.init_process_group first')
        raise e
    world_size = torch.distributed.get_world_size()
    tensor = torch.zeros(world_size * N_per_process, dtype=torch.int).cuda()
    if rank == 0:
        device_ids = get_available_GPUs(world_size * N_per_process)
        tensor = torch.tensor(device_ids, dtype=torch.int).cuda()
    torch.distributed.broadcast(tensor, 0)
    ids = list(tensor.cpu().numpy())
    return ids[N_per_process * rank : N_per_process * rank + N_per_process]