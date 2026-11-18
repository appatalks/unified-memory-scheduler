import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/benchmark/vrambridge-prefetch-screen.py'
spec = importlib.util.spec_from_file_location('prefetch_screen', SCRIPT)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
wrapper_spec = importlib.util.spec_from_file_location(
    'prepare_wrapper', SCRIPT.with_name('vrambridge-prepare-wrapper.py'))
wrapper = importlib.util.module_from_spec(wrapper_spec)
wrapper_spec.loader.exec_module(wrapper)


class PrefetchScreenTests(unittest.TestCase):
    def test_prepare_wrapper_execs_sibling_with_explicit_opt_in(self):
        expected = str(Path(wrapper.__file__).resolve().with_name('llama-server'))
        with mock.patch.object(sys, 'argv', ['wrapper', '--model', 'test.gguf']), \
                mock.patch.object(wrapper.os, 'execv') as execute, \
                mock.patch.dict(wrapper.os.environ, {}, clear=False):
            wrapper.main()
            self.assertEqual(wrapper.os.environ['VRAMBRIDGE_HYBRID_CPU_PREPARE'], '1')
            execute.assert_called_once_with(expected, [expected, '--model', 'test.gguf'])

    def test_paired_runs_validate_marker_identity_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ('baseline', 'candidate', 'probe'):
                path = root / kind
                path.mkdir()
                for name in ('llama-bench', 'libllama.so.0.7.0', 'libggml-base.so.0.27.0', 'libggml-cuda.so'):
                    (path / name).write_bytes(name.encode())
                (path / 'libggml-cpu.so').write_bytes(kind.encode())
            model = root / 'model.gguf'
            model.write_bytes(b'GGUF')
            output = root / 'result.json'
            order = []

            def run(args, **kwargs):
                if args[0] == 'nvidia-smi':
                    return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
                kind = Path(args[0]).parent.name
                order.append(kind)
                self.assertNotIn('GGML_FAKE_OVERRIDE', kwargs['env'])
                token_count = int(args[args.index('-n') + 1])
                row = {'n_gen': token_count, 'n_prompt': 0, 'n_cpu_moe': 99, 'n_gpu_layers': 99,
                       'n_threads': 8, 'type_k': 'f16', 'type_v': 'f16', 'backends': 'CUDA',
                       'samples_ts': [10.0 if kind == 'baseline' else 11.0], 'model_type': 'test'}
                log = ('loaded CPU backend from ' + str(root / kind / 'libggml-cpu.so') + '\n' +
                       'loaded CUDA backend from ' + str(root / kind / 'libggml-cuda.so') + '\n' +
                       ('VRAMBRIDGE_NTA_ACTIVE q4_k\n' if kind == 'probe' else ''))
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([row]), stderr=log)

            args = ['screen', '--baseline', str(root / 'baseline'), '--candidate', str(root / 'candidate'),
                    '--probe', str(root / 'probe'), '--model', str(model), '--marker', 'q4_k',
                    '--tokens', '16', '--repetitions', '2', '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(screen.subprocess, 'run', side_effect=run), \
                    mock.patch.dict(screen.os.environ, {'GGML_FAKE_OVERRIDE': '1'}):
                screen.main()
            data = json.loads(output.read_text())
            self.assertEqual(order, ['probe', 'baseline', 'candidate', 'candidate', 'baseline'])
            self.assertEqual(data['status'], 'complete')
            self.assertEqual(data['activation_verified'], 'q4_k')
            self.assertEqual(data['medians'], {'baseline': 10.0, 'candidate': 11.0})
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_busy_gpu_is_rejected(self):
        result = subprocess.CompletedProcess([], 0, stdout='1234\n', stderr='')
        with mock.patch.object(screen.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'GPU is in use'):
                screen.gpu_idle()

    def test_missing_activation_is_not_timed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kind in ('baseline', 'candidate', 'probe'):
                path = root / kind
                path.mkdir()
                for name in ('llama-bench', 'libllama.so.0.7.0', 'libggml-base.so.0.27.0', 'libggml-cuda.so'):
                    (path / name).write_bytes(name.encode())
                (path / 'libggml-cpu.so').write_bytes(kind.encode())
            model, output = root / 'model.gguf', root / 'result.json'
            model.write_bytes(b'GGUF')

            def run(args, **kwargs):
                if args[0] == 'nvidia-smi':
                    return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
                row = {'n_gen': 16, 'n_prompt': 0, 'n_cpu_moe': 99, 'n_gpu_layers': 99,
                       'n_threads': 8, 'type_k': 'f16', 'type_v': 'f16', 'backends': 'CUDA'}
                log = ('loaded CPU backend from ' + str(root / 'probe/libggml-cpu.so') + '\n' +
                       'loaded CUDA backend from ' + str(root / 'probe/libggml-cuda.so'))
                return subprocess.CompletedProcess(args, 0, stdout=json.dumps([row]), stderr=log)

            args = ['screen', '--baseline', str(root / 'baseline'), '--candidate', str(root / 'candidate'),
                    '--probe', str(root / 'probe'), '--model', str(model), '--marker', 'q4_k',
                    '--tokens', '16', '--repetitions', '1', '--output', str(output)]
            with mock.patch.object(sys, 'argv', args), mock.patch.object(screen.subprocess, 'run', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'did not activate'):
                    screen.main()
            data = json.loads(output.read_text())
            self.assertEqual(data['status'], 'failed')
            self.assertEqual(data['samples'], [])


if __name__ == '__main__':
    unittest.main()
