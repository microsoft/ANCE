import numpy as np
import json
import copy
import torch
import sys
import random
import argparse
import pickle
import glob
import os

from typing import List, Set, Dict, Tuple, Callable, Iterable, Any
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset, get_worker_info
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class SimplifiedStreamingDataset(IterableDataset):
    # ix_func determines whether this worker processes this line: f(i, local_rank, world_size, record) -> bool
    def __init__(self, records, fn, ix_func:Callable[[int, int, int, Any], bool]=None):
        super().__init__()
        self.num_replicas=-1
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")

        self.records = records
        self.fn = fn
        self.ix_func = ix_func
        if self.ix_func is None:
            def default_ix_funct(i, local_rank, world_size, record):
                return i % world_size == local_rank
            self.ix_func = default_ix_funct
    
    def __iter__(self):
        try:
            self.records.seek(0)
        except:
            pass
        for i, record in enumerate(self.records):
            if self.num_replicas != -1 and not self.ix_func(i, self.rank, self.num_replicas, record):
                continue
            rows = self.fn(record, i)
            for rec in rows:
                yield rec


class StreamingDataset(IterableDataset):
    """Deprecated. use SimplifiedStreamingDataset."""
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn

    def gen_record(self, buffer):
        for b_line, b_i in buffer:
            records = self.fn(b_line, b_i)
            for rec in records:
                yield rec
    
    def __iter__(self):
        worker_info = get_worker_info()
        buffer = []
        for i, element in enumerate(self.elements):
            if worker_info is not None:
                if worker_info.id!= i % worker_info.num_workers:
                    continue
            buffer.append(element)
        yield from self.gen_record(buffer)
    

class StreamingDataLoader:
    """Deprecated. use stock DataLoader with StreamingDataset."""
    def __init__(self, records:Iterable[Any], fn: Callable[[Any, int], List[Any]], batch_size:int, num_workers:int, get_all:bool=False):
        if not dist.is_available():
            print("Distribution mode not available")
        # records is an iterable
        self.records = records
        self.batch_size = batch_size
        self.num_workers = num_workers
        # fn: record, i -> [output] 
        self.fn = fn
        self.capacity = batch_size*100 #stream 100 batches at a time
        self.num_replicas=-1
        self.dataloader = None
        self.get_all = get_all 

    def gen_dataloader(self):
        if len(self.record_container)>0:
            chunk = StreamingDataset(self.record_container, self.fn)
            self.dataloader = iter(DataLoader(chunk, self.batch_size, num_workers=self.num_workers, pin_memory=True))
        # self.batch_container = [batch for batch in dataloader]

    def __iter__(self):
        try:
            self.records.seek(0) # if file-like, move pointer to beginning
        except:
            pass
        self.end_of_records = False
        self.record_iter = iter(self.records)
        self.record_container = []
        self.batch_container = []
        self.record_num = -1  
        self.num_replicas = -1
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        return self

    def populate(self):
        while len(self.record_container)<self.capacity:
            try:
                record = next(self.record_iter)
            except StopIteration:
                self.end_of_records = True
                break
            self.record_num+=1
            if not self.get_all and self.num_replicas!=-1 and self.record_num%self.num_replicas!=self.rank:
                continue
            self.record_container.append([record, self.record_num])
        self.gen_dataloader()
        
    def __next__(self):
        # using next is subtle -the program doesnt crash if there are any exceptions below
        if self.dataloader is None:
            self.populate()

        if self.dataloader is None:
            raise StopIteration

        try:
            batch = next(self.dataloader)
            return batch
        except StopIteration:
            self.dataloader = None
            self.record_container = []
            if self.end_of_records:
                raise StopIteration
            self.populate()
            batch = next(self.dataloader)
            return batch

class CachedStreamingDataLoader:
    def __init__(self, records:Iterable[Any], fn: Callable[[Any, int], List[Any]], batch_size:int, cache_dir:str, prefix:str="tmp", ix_func:Callable[[int, int, int, Any], bool]=None):
        if not dist.is_available():
            print("Distribution mode not available")
        ssd = SimplifiedStreamingDataset(records, fn, ix_func)
        self.dl = DataLoader(ssd, batch_size, num_workers=1)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.rank==0 and cache_dir is not None and not os.path.exists(cache_dir):
            print("{0} doesn't exist. Creating...".format(cache_dir))
            os.makedirs(cache_dir)
        if dist.is_initialized():
            dist.barrier()
        self.cache_dir = cache_dir
        self.batches_per_file = 100
        self.prefix = prefix
        self.is_first_iter = True
        glob_pattern = "{0}_rank{1}_part*.pb".format(self.prefix, str(self.rank))
        if self.cache_dir is not None:
            self.glob_pattern = os.path.join(self.cache_dir, glob_pattern)

    def first_iter(self):
        # TODO: cleanup folder before first iter
        container = []
        c = 0
        for i, batch in enumerate(self.dl):
            if self.cache_dir is not None:
                if len(container)>=self.batches_per_file:
                    save_path = os.path.join(self.cache_dir, "{0}_rank{1}_part{2}.pb".format(self.prefix, str(self.rank), str(c)))
                    torch.save(container, save_path)
                    container = []
                    c+=1
                container.append(batch)
            yield batch
        if len(container)>0 and self.cache_dir is not None:
            save_path = os.path.join(self.cache_dir, "{0}_rank{1}_part{2}.pb".format(self.prefix, str(self.rank), str(c)))
            torch.save(container, save_path)

    def cached_iter(self):
        print("Load from cache: {0}".format(self.glob_pattern))
        for f in glob.glob(self.glob_pattern):
            container = torch.load(f)
            for batch in container:
                yield batch

    def cleanup(self):
        print("Deleting from cache: {0}".format(self.glob_pattern))
        for f in glob.glob(self.glob_pattern):
            os.remove(f)
    
    def __call__(self, is_first_iter):
        self.is_first_iter = is_first_iter
        return self

    def __iter__(self):
        if self.is_first_iter:
            yield from self.first_iter()
        else:
            yield from self.cached_iter()

# cribbed from https://github.com/facebookresearch/maskrcnn-benchmark/blob/24c8c90efdb7cc51381af5ce0205b23567c3cd21/maskrcnn_benchmark/utils/comm.py#L48-L88
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list 

def all_gather_cpu(data, prefix="tmp", cleanup=True):
    """
    Pickles data from all processes, writes to disk, then read all of them
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # serialized to a Tensor
    pathholder = "{0}_part_{1}"
    self_pkl_path = pathholder.format(prefix, str(rank))
    with open(self_pkl_path, "wb") as handle:
        pickle.dump(data, handle, protocol=4)
    dist.barrier()
    data_list = []
    for i in range(world_size):
        if i!=rank:
            pkl_path = pathholder.format(prefix, str(i))
            with open(pkl_path, "rb") as handle:
                d = pickle.load(handle)
            data_list.append(d)
        else:
            data_list.append(data)
    dist.barrier()
    if cleanup:
        os.remove(self_pkl_path)
    dist.barrier()
    return data_list
    
def test_ssd(args):
    samples = range(50)
    ssd = SimplifiedStreamingDataset(samples, lambda x, i: [x])
    dl = DataLoader(ssd, batch_size=4, num_workers=2)
    # simulate 3 epochs
    for _ in range(3):
        for batch in dl:
            print(batch)
        if dist.is_initialized():
            dist.barrier()

if __name__=="__main__":
    # run with python3 -m torch.distributed.launch --nproc_per_node=2 iterable_dataset.py to test on single node
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    if args.local_rank!=-1:
        torch.distributed.init_process_group(backend='nccl')
    test_ssd(args)