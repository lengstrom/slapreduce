# slapreduce: simple mapreduce for slurm

### Install:
```
pip install -e git@github.com:lengstrom/slapreduce.git
```

### Example:
```python
from slapreduce import slap, collect

xs = range(10)
fn = lambda x: {'wow':x ** 2, 'such': x ** 3, 'map': x ** 4}

dired_out = '/mnt/xfs/home/engstrom/scratch/slapreduce_test'
# define args
gres = {
    'exclude': r'deep-chungus-[1-6]',
    'gres': 'gpu:a100:1',
    'cpus_per_task':2,
    'nodes': 1,
}

partition = 'background'
job_name = 'slapreduce_test_v0'

# "map" (can set block to false if you dont want to synchonously wait)
slap(fn, [{'x': x} for x in xs], dired_out, gres, partition=partition,
        job_name=job_name, block=True)

# "reduce" once jobs are done
for kw, ret in collect(dired_out):
    print(kw, ret)
```

Output:
```
👋 slapreduce: Slapping in directory: /mnt/xfs/home/engstrom/scratch/slapreduce_test
👋 slapreduce: Logs file: /mnt/xfs/home/engstrom/scratch/slapreduce_test/logs
👋 slapreduce: num jobs:  10
👋 slapreduce: first job: {'x': 0}
👋 slapreduce: last job: {'x': 9}
{'x': 0} {'wow': 0, 'such': 0, 'map': 0}
{'x': 1} {'wow': 1, 'such': 1, 'map': 1}
{'x': 2} {'wow': 4, 'such': 8, 'map': 16}
{'x': 3} {'wow': 9, 'such': 27, 'map': 81}
{'x': 4} {'wow': 16, 'such': 64, 'map': 256}
{'x': 5} {'wow': 25, 'such': 125, 'map': 625}
{'x': 6} {'wow': 36, 'such': 216, 'map': 1296}
{'x': 7} {'wow': 49, 'such': 343, 'map': 2401}
{'x': 8} {'wow': 64, 'such': 512, 'map': 4096}
{'x': 9} {'wow': 81, 'such': 729, 'map': 6561}
```
