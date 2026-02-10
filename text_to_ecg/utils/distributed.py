"""
Distributed training utilities.

Supports multiple backends:
- Single GPU (default)
- DeepSpeed
- Horovod
"""

import torch
from abc import ABC, abstractmethod


# Global state
is_distributed = False
backend = None


class DistributedBackend(ABC):
    """Abstract base class for distributed backends."""

    @abstractmethod
    def initialize(self):
        """Initialize the distributed backend."""
        pass

    @abstractmethod
    def get_world_size(self):
        """Get total number of processes."""
        pass

    @abstractmethod
    def get_rank(self):
        """Get rank of current process."""
        pass

    @abstractmethod
    def is_root_worker(self):
        """Check if this is the root worker."""
        pass

    @abstractmethod
    def is_local_root_worker(self):
        """Check if this is the local root worker."""
        pass

    @abstractmethod
    def local_barrier(self):
        """Synchronize local processes."""
        pass

    @abstractmethod
    def distribute(self, args, model, optimizer, model_parameters,
                   training_data, lr_scheduler, config_params):
        """Set up distributed training."""
        pass

    @abstractmethod
    def average_all(self, tensor):
        """Average tensor across all processes."""
        pass

    @abstractmethod
    def check_batch_size(self, batch_size):
        """Validate batch size for distributed training."""
        pass


class DummyBackend(DistributedBackend):
    """Dummy backend for single GPU training."""

    def initialize(self):
        pass

    def get_world_size(self):
        return 1

    def get_rank(self):
        return 0

    def is_root_worker(self):
        return True

    def is_local_root_worker(self):
        return True

    def local_barrier(self):
        pass

    def distribute(self, args, model, optimizer, model_parameters,
                   training_data, lr_scheduler, config_params):
        from torch.utils.data import DataLoader
        if isinstance(training_data, DataLoader):
            dataloader = training_data
        else:
            dataloader = DataLoader(training_data, batch_size=config_params.get('train_batch_size', 32))
        return model, optimizer, dataloader, lr_scheduler

    def average_all(self, tensor):
        return tensor

    def check_batch_size(self, batch_size):
        pass


class DeepSpeedBackend(DistributedBackend):
    """DeepSpeed backend for distributed training."""

    def __init__(self):
        self.backend_module = None
        self._initialized = False

    def initialize(self):
        try:
            import deepspeed
            self.backend_module = deepspeed
            deepspeed.init_distributed()
            self._initialized = True
        except ImportError:
            raise ImportError("DeepSpeed is not installed. Run: pip install deepspeed")

    def get_world_size(self):
        import torch.distributed as dist
        return dist.get_world_size()

    def get_rank(self):
        import torch.distributed as dist
        return dist.get_rank()

    def is_root_worker(self):
        return self.get_rank() == 0

    def is_local_root_worker(self):
        return self.get_rank() == 0

    def local_barrier(self):
        import torch.distributed as dist
        dist.barrier()

    def distribute(self, args, model, optimizer, model_parameters,
                   training_data, lr_scheduler, config_params):
        model_engine, optimizer, dataloader, lr_scheduler = self.backend_module.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            config_params=config_params,
        )
        return model_engine, optimizer, dataloader, lr_scheduler

    def average_all(self, tensor):
        import torch.distributed as dist
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.get_world_size()
        return tensor

    def check_batch_size(self, batch_size):
        world_size = self.get_world_size()
        assert batch_size % world_size == 0, \
            f"Batch size {batch_size} must be divisible by world size {world_size}"


class HorovodBackend(DistributedBackend):
    """Horovod backend for distributed training."""

    def __init__(self):
        self.backend_module = None
        self._initialized = False

    def initialize(self):
        try:
            import horovod.torch as hvd
            self.backend_module = hvd
            hvd.init()
            self._initialized = True
        except ImportError:
            raise ImportError("Horovod is not installed. Run: pip install horovod")

    def get_world_size(self):
        return self.backend_module.size()

    def get_rank(self):
        return self.backend_module.rank()

    def is_root_worker(self):
        return self.get_rank() == 0

    def is_local_root_worker(self):
        return self.backend_module.local_rank() == 0

    def local_barrier(self):
        self.backend_module.allreduce(torch.tensor(0), name='barrier')

    def distribute(self, args, model, optimizer, model_parameters,
                   training_data, lr_scheduler, config_params):
        hvd = self.backend_module
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters()
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            training_data,
            num_replicas=hvd.size(),
            rank=hvd.rank()
        )
        dataloader = DataLoader(
            training_data,
            batch_size=config_params.get('train_batch_size', 32),
            sampler=sampler
        )
        return model, optimizer, dataloader, lr_scheduler

    def average_all(self, tensor):
        return self.backend_module.allreduce(tensor)

    def check_batch_size(self, batch_size):
        pass


def set_backend_from_args(args):
    """Set distributed backend based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Initialized backend instance
    """
    global is_distributed, backend

    if hasattr(args, 'deepspeed') and args.deepspeed:
        backend = DeepSpeedBackend()
        is_distributed = True
    elif hasattr(args, 'horovod') and args.horovod:
        backend = HorovodBackend()
        is_distributed = True
    else:
        backend = DummyBackend()
        is_distributed = False

    return backend


def using_backend(backend_class):
    """Check if using a specific backend.

    Args:
        backend_class: Backend class to check

    Returns:
        True if using the specified backend
    """
    global backend
    return isinstance(backend, backend_class)


def wrap_arg_parser(parser):
    """Add distributed training arguments to parser.

    Args:
        parser: ArgumentParser instance

    Returns:
        Modified parser
    """
    group = parser.add_argument_group('Distributed training')
    group.add_argument('--deepspeed', action='store_true',
                       help='Use DeepSpeed for distributed training')
    group.add_argument('--horovod', action='store_true',
                       help='Use Horovod for distributed training')
    group.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    return parser
