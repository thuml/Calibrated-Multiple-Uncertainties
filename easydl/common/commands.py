__package__ = 'easydl.common'


def runTask():
    """
    usage: runTask [-h] [--maxGPU MAXGPU] [--needGPU NEEDGPU] [--maxLoad MAXLOAD]
                   [--maxMemory MAXMEMORY] [--sleeptime SLEEPTIME] [--user USER]
                   file

    This is a command. After installing the package, you can get the command in commandline.
    The command is designed to run tasks automatically. Suppose that one has 10
    tasks to run but only has 8 GPUs, each task requires 1 GPU. He has to run 8
    tasks first, then frequently checks whether tasks finish, which is tedious.
    With this commoand ``runTask``, one can specify 10 commands to run in a file.
    Every ``sleeptime`` seconds, it checks whether ``needGPU`` GPUs are available.
    If there are enough GPUs to run the task, it gets one line from ``file`` and executes
    the line. (If it succeeds in executing the line, that line will be removed.)
    The ``user`` argument is needed to query how many GPUs are used by the specified user (together
    with ``maxGPU``, it limits the number of GPUs one can use. This is often the
    case when GPUs are shared and one can't take up all GPUs)

    positional arguments:
      file                  file that contains one task per line (should end with
                            & to be run background. If not, & will be appended automatically.)

    optional arguments:
      -h, --help            show this help message and exit
      --maxGPU MAXGPU       maximum GPU to use by one user (default: 100)
      --needGPU NEEDGPU     number of GPUs per task/line (default: 1)
      --maxLoad MAXLOAD     GPU with load larger than this will be regarded as not
                            available (default: 0.1)
      --maxMemory MAXMEMORY
                            GPU with memory usage larger than this will be
                            regarded as not available (default: 0.1)
      --sleeptime SLEEPTIME
                            sleep time after executing one task/line (in seconds)
                            (default: 180.0)
      --user USER           query how many GPUs user used so that it does not
                            violatethe limitation of maxGPU per user (default:
                            None)

    """

    help_msg = '''
    This is a command. After installing the package, you can get the command in commandline.
    The command is designed to run tasks automatically. Suppose that one has 10
    tasks to run but only has 8 GPUs, each task requires 1 GPU. He has to run 8
    tasks first, then frequently checks whether tasks finish, which is tedious.
    With this commoand ``runTask``, one can specify 10 commands to run in a file.
    Every ``sleeptime`` seconds, it checks whether ``needGPU`` GPUs are available.
    If there are enough GPUs to run the task, it gets one line from ``file`` and executes
    the line. (If it succeeds in executing the line, that line will be removed.)
    The ``user`` argument is needed to query how many GPUs are used by the specified user (together
    with ``maxGPU``, it limits the number of GPUs one can use. This is often the
    case when GPUs are shared and one can't take up all GPUs)
    '''
    import argparse
    parser = argparse.ArgumentParser(prog='runTask', formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=help_msg)
    parser.add_argument('file', nargs=1, help='file that contains one task per line (should end with & to be run background. If not, & will be appended automatically.)')
    parser.add_argument('--maxGPU', type=int, default=100, help='maximum GPU to use by one user')
    parser.add_argument('--needGPU', type=int, default=1, help='number of GPUs per task/line')
    parser.add_argument('--maxLoad', type=float, default=0.1, help='GPU with load larger than this will be regarded as not available')
    parser.add_argument('--maxMemory', type=float, default=0.1, help='GPU with memory usage larger than this will be regarded as not available')
    parser.add_argument('--sleeptime', type=float, default=180.0, help='sleep time after executing one task/line (in seconds)')
    parser.add_argument('--user', required=True, type=str, help='query how many GPUs user used so that it does not violatethe limitation of maxGPU per user')
    args = parser.parse_args()

    maxGPU = args.maxGPU
    needGPU = args.needGPU
    maxLoad = args.maxLoad
    maxMemory = args.maxMemory
    file = args.file[0]
    user = args.user
    sleeptime = args.sleeptime
    from .gpuutils import select_GPUs
    from subprocess import Popen, PIPE
    import time

    import random
    import os

    while True:
        with open(file) as f:
            lines = [line for line in f if line.strip()]
        if lines:
            while True:
                s = 'for x in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do ps -p $x -o pid,user | grep "%s"; done' % user
                p = Popen(s, stdout=PIPE, shell=True)
                ans = p.stdout.read()
                mygpu = len(ans.splitlines())
                deviceIDs = []
                try:
                    deviceIDs = select_GPUs(N_per_process=needGPU, max_utilization=maxLoad,max_memory_usage=maxMemory)
                except Exception as e:
                    deviceIDs = []
                find = False
                if mygpu < maxGPU and len(deviceIDs) >= needGPU:
                    command = lines[0].strip()
                    if not command.endswith('&'):
                        command += ' &'
                    os.system(command)
                    print('runing command(%s)' % command)
                    find = True
                time.sleep(sleeptime)
                if find:
                    break
            with open(file, 'w') as f:
                for line in lines[1:]:
                    f.write(line)
        else:
            break