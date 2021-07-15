from dataclasses import dataclass, field
import psutil
import torch

@dataclass
class Dev:
    """
    Contains data about the computing device used to establish computations and
    parallelism. 
    The memory_free sum takes into account cleared cache, giving torch.cuda does
    not release the memory when it deallocates the object.
    """
    name: str                               #Device name (cpu, cuda:0,1...)
    n_cores : int = 1                       #N of cores
    memory : int = field(init=False)        #Total memory size
    memory_free : int = field(init=False)   #Free memory (Total - (reserved - unused))
    device_type: str = 'undefined'          #CPU,GPU

    def __post_init__(self):
        self.memory = torch.cuda.get_device_properties(self.name).total_memory if self.device_type=='gpu' else psutil.virtual_memory()[0]  #Total memory
        self.memory_free = self.mem_update()
    def mem_update(self):
        """
        Updates memory_free with the current memory available. Note that it can not take into account
        memory used by other processes outside of pytorch for cuda devices.
        """
        self.memory_free = (self.memory - (torch.cuda.memory_reserved(self.name) - torch.cuda.memory_allocated(self.name))
                                         ) if self.device_type=='gpu' else psutil.virtual_memory()[1]
        return self.memory_free

if __name__ == "__main__":
    """
    For testing purposes
    """
    test = [Dev(name="cuda:0", device_type="gpu"), Dev(name="cpu", n_cores=psutil.cpu_count(), device_type="cpu")]
    for i in test:
        print(f"Test on {i.name}:")
        print(i)