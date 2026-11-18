#!/usr/bin/env python3
"""Screen matched CPU prefetch plugins with isolated native llama-bench runs."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def gpu_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('GPU is in use by another process; refusing to run')


def main():
    p = argparse.ArgumentParser()
    for arg in ('baseline', 'candidate', 'probe', 'model', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--marker', required=True, choices=('q4_k', 'iq3_s', 'mxfp4'))
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--repetitions', type=int, default=3)
    a = p.parse_args()
    temporary = a.output.with_suffix(a.output.suffix + '.tmp')
    if a.output.exists() or temporary.exists() or not 16 <= a.tokens <= 4096 or not 1 <= a.repetitions <= 10:
        raise ValueError('output must be new; use 16-4096 tokens and 1-10 repetitions')
    model = a.model.resolve(strict=True)
    paths = {key: getattr(a, key).resolve(strict=True) for key in ('baseline', 'candidate', 'probe')}
    for key, path in paths.items():
        if not (path / 'llama-bench').is_file() or not (path / 'libggml-cpu.so').is_file():
            raise ValueError('runtime is incomplete: ' + key)
    for filename in ('llama-bench', 'libllama.so.0.7.0', 'libggml-base.so.0.27.0', 'libggml-cuda.so'):
        if any(sha(paths['baseline'] / filename) != sha(paths[kind] / filename)
               for kind in ('candidate', 'probe')):
            raise ValueError('non-CPU component differs: ' + filename)
    if sha(paths['baseline'] / 'libggml-cpu.so') == sha(paths['candidate'] / 'libggml-cpu.so'):
        raise ValueError('candidate CPU plugin is identical to control')
    data = {'schema_version': 1, 'status': 'running', 'model_name': model.name,
            'model_size': model.stat().st_size, 'marker': a.marker, 'samples': [],
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'tokens': a.tokens, 'repetitions': a.repetitions, 'threads': 8,
            'cpu_moe_layers': 99, 'gpu_layers': 99,
            'identity': {k: {'executable': sha(v / 'llama-bench'), 'cpu': sha(v / 'libggml-cpu.so'),
                             'cuda': sha(v / 'libggml-cuda.so')} for k, v in paths.items()},
            'limitations': ['llama-bench synthetic token workload; not a natural request',
                            'native static expert placement, not adaptive hybrid or full logits',
                            'one process/load per sample; no cache eviction',
                            'kernel path instrumentation is separate and not timed',
                            'existing mmap pages may be warm; load time is not measured']}
    def save():
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as out:
            out.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, a.output)
    def run(kind, tokens, label):
        gpu_idle()
        binary = paths[kind] / 'llama-bench'
        command = [str(binary), '-m', str(model), '-p', '0', '-n', str(tokens), '-r', '1',
                   '-t', '8', '-ngl', '99', '-ncmoe', '99', '-o', 'json']
        started = time.monotonic()
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('GGML_', 'LLAMA_', 'VRAMBRIDGE_', 'STRATA_', 'LD_'))}
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200, env=environment)
        log_path = a.output.parent / (a.output.stem + '-' + label + '.log')
        fd = os.open(log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as log:
            log.write(result.stderr)
        gpu_idle()
        if result.returncode:
            raise RuntimeError(f'{label} failed, see private log; return code {result.returncode}')
        expected = 'loaded CPU backend from ' + str(paths[kind] / 'libggml-cpu.so')
        expected_cuda = 'loaded CUDA backend from ' + str(paths[kind] / 'libggml-cuda.so')
        if expected not in result.stderr or expected_cuda not in result.stderr:
            raise RuntimeError('runtime failed backend-identity check: ' + label)
        entries = json.loads(result.stdout)
        if (len(entries) != 1 or entries[0]['n_gen'] != tokens or entries[0]['n_prompt'] != 0 or
                entries[0]['n_cpu_moe'] != 99 or entries[0]['n_gpu_layers'] != 99 or
                entries[0]['n_threads'] != 8 or entries[0]['type_k'] != 'f16' or
                entries[0]['type_v'] != 'f16' or 'CUDA' not in entries[0]['backends']):
            raise RuntimeError('unexpected benchmark test or placement: ' + label)
        return entries[0], result.stderr, time.monotonic() - started
    try:
        probe, log, _ = run('probe', 16, 'probe')
        if 'VRAMBRIDGE_NTA_ACTIVE ' + a.marker not in log:
            raise RuntimeError('experimental kernel did not activate in the instrumented run')
        data['activation_verified'] = a.marker
        save()
        for i in range(a.repetitions):
            for kind in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
                row, _, wall = run(kind, a.tokens, f'{i}-{kind}')
                if (len(row['samples_ts']) != 1 or not math.isfinite(row['samples_ts'][0]) or
                        row['samples_ts'][0] <= 0):
                    raise RuntimeError('invalid single-run measurement')
                data['samples'].append({'repetition': i, 'arm': kind, 'tok_s': row['samples_ts'][0],
                                        'model_type': row['model_type'], 'wall_seconds': wall,
                                        'reported_backends': row['backends']})
                save()
        data['medians'] = {kind: statistics.median(x['tok_s'] for x in data['samples'] if x['arm'] == kind)
                           for kind in ('baseline', 'candidate')}
        data['paired_change_percent'] = [100 * (next(x['tok_s'] for x in data['samples'] if x['arm'] == 'candidate' and x['repetition'] == i) /
                                              next(x['tok_s'] for x in data['samples'] if x['arm'] == 'baseline' and x['repetition'] == i) - 1)
                                         for i in range(a.repetitions)]
        data['status'] = 'complete'
    except Exception as e:
        data['status'] = 'failed'
        data['error'] = str(e)
        save()
        raise
    save()
    print(json.dumps({'model': data['model_name'], 'medians': data['medians'],
                      'paired_change_percent': data['paired_change_percent']}))


if __name__ == '__main__':
    main()
