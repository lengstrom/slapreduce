import submitit
import os
import json
import time
import sys
from pathlib import Path
import shutil
from functools import partial
import types
import dill as pickle

def f_wrapper(f, out_path, **kwargs):
    loader = JAX_LOADER_HANDLER

    if not out_path.exists():
        ret = f(**kwargs)
        loader.save((kwargs, ret), out_path)

def path_for_job_i(dired_out, i):
    # back compat
    putative = dired_out / f'{i}.pkl'
    if not putative.exists():
        compat_putative = dired_out / f'{i}.npy'
        if compat_putative.exists():
            return compat_putative

    return putative

def jax_loader(p):
    with open(p, 'rb') as fp:
        return pickle.load(fp)

def jax_saver(x, p):
    with open(p, 'wb') as fp:
        pickle.dump(x, fp)

JAX_LOADER_HANDLER = types.SimpleNamespace()
JAX_LOADER_HANDLER.load = jax_loader
JAX_LOADER_HANDLER.save = jax_saver

def slapmsg(msg, *args):
    msg = ' '.join([msg] + list(args))
    return f'>> slapreduce 👋: {msg}'

def slap(*args, **kwargs):
    def get_env_value(name, is_bool):
        if not name in os.environ:
            return False

        value = os.environ[name]
        if value == '':
            raise ValueError(slapmsg(f'Environment variable {name} is empty'))  

        if is_bool:
            return bool(int(value))

        return value

    if 'NAME' in os.environ:
        kwargs['job_name'] = get_env_value('NAME', is_bool=False)
    
    if 'PARTITION' in os.environ:
        kwargs['partition'] = get_env_value('PARTITION', is_bool=False)

    if 'DRY_RUN' in os.environ:
        kwargs['dry_run'] = get_env_value('DRY_RUN', is_bool=True)

    base_slap(*args, **kwargs)

def base_slap(f, xs, dired_out, gres, *, partition, job_name, dry_run=False,
              debug=False, maxjobs=1000, block=True, timeout_days=3):
    '''
    f: callable that you want to serialize on cluster. can return anything.
    xs: list of args, kwargs for f
    dired_out: directory to save outputs
    gres: slurm gres dictionary
    debug: run one job locally..
    chunksize: int, max jobs at once
    '''

    dired_out = Path(dired_out)
    jobs = []

    assert dired_out.parent.exists()
    logs_dired = dired_out / 'logs'
    logs_dired.mkdir(exist_ok=True, parents=True)

    metadata = {
        'num_jobs': len(xs),
    }

    metadata_file = dired_out / 'metadata.json'
    if metadata_file.exists():
        metadata_on_disk = json.load(open(metadata_file, 'r'))
        assert metadata_on_disk == metadata

    with open(metadata_file, 'w') as fp:
        json.dump(metadata, fp)

    print('*' * 80)
    print(slapmsg('Mapping in directory:', dired_out))
    print(slapmsg('Logs file:', logs_dired))
    print(slapmsg('num jobs: ', len(xs)))
    print(slapmsg('first job:', xs[0]))
    print(slapmsg('last job:', xs[-1]))
    if dry_run:
        sys.exit(0)

    timeout_min = int(60 * 24 * timeout_days)
    executor = submitit.AutoExecutor(logs_dired)
    executor.update_parameters(slurm_partition=partition,
                               timeout_min=timeout_min,
                               slurm_additional_parameters=gres)

    executor.update_parameters(name=job_name)
    executor.update_parameters(slurm_array_parallelism=maxjobs)
    callables = []
    xs = list(enumerate(xs))
    xs = sorted(xs, key=lambda x: x[1].get('epochs', 1))
    for i, kwargs in xs:
        job_path = path_for_job_i(dired_out, i)
        if not job_path.exists():
            callable = partial(f_wrapper, f=f, out_path=job_path, **kwargs)
            if not debug:
                callables.append(callable)
            else:
                callable()

    jobs = executor.submit_array(callables)

    if block:
        for j in jobs:
            j.result()

def collect(dired_out, strict=False, warn=True, with_paths=False):
    dired_out = Path(dired_out)

    with open(dired_out / 'metadata.json', 'r') as fp:
        metadata = json.load(fp)

    num_jobs = metadata['num_jobs']
    loader = JAX_LOADER_HANDLER

    for i in range(num_jobs):
        job_path = path_for_job_i(dired_out, i)
        if not job_path.exists():
            if strict:
                raise ValueError(slapmsg(f'Job {i} not found at {job_path}'))
            else:
                if warn:
                    print(slapmsg(f'Job {i} not found at {job_path}; skipping...'))

                continue

        kw, ret = loader.load(job_path)
        if not with_paths:
            yield kw, ret
        else:
            yield job_path, kw, ret
