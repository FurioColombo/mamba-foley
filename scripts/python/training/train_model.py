import os
import sys
import subprocess

root_folder_path = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], universal_newlines=True).strip())
modules_folder_path = os.path.join(root_folder_path, "modules")
sys.path.append(root_folder_path)
sys.path.append(modules_folder_path)
from utils.file_system import ProjectPaths
from config.config import Config

# set gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = Config.get_config(ProjectPaths.config_script_train).CUDA_VISIBLE_DEVICES

from torch.cuda import device_count
from torch.multiprocessing import spawn

from modules.train.learner import train, train_distributed

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]

def check_model_config():
    config = Config.get_config(ProjectPaths.config_script_train)
    # DATASET
    if config.data.train_cond_dirs is not None:
        print('Conditioning will be loaded from file!')

    if config.profiler.use_profiler is True:
        assert config.profiler.wait is not None
        assert config.profiler.warmup is not None
        assert config.profiler.active is not None
    # GPUs
    print("Cuda GPUs codes set to be used:", config.CUDA_VISIBLE_DEVICES)

def main():
    config = Config.get_config(ProjectPaths.config_script_train)
    check_model_config()
    replica_count = device_count()

    if replica_count > 1:
        if config.training.batch_size % replica_count != 0:
            raise ValueError(
                f"Batch size {config.training.batch_size} is not evenly divisible by # GPUs {replica_count}."
            )
        config.training.batch_size = config.training.batch_size // replica_count
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, config),
            nprocs=replica_count,
            join=True,
        )

    else:
        train(config)

if __name__ == "__main__": 
    main()
