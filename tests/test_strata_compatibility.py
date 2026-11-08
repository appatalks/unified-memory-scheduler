import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import shutil
import sys
import tempfile
import unittest
from contextlib import nullcontext


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "strata_compatibility", ROOT / "tools/benchmark/vrambridge-strata-compatibility.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
bridge_spec = importlib.util.spec_from_file_location(
    "strata_server", ROOT / "tools/benchmark/vrambridge-strata-server.py")
bridge = importlib.util.module_from_spec(bridge_spec)
bridge_spec.loader.exec_module(bridge)
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class CaptureTests(unittest.TestCase):
    def test_protocol_heartbeats_are_not_counted_as_tokens(self):
        self.assertEqual(check.collect_tokens(iter([None, 17, None, 18])), [17, 18])
        with self.assertRaises(ValueError):
            check.collect_tokens(iter(["19"]))

    @unittest.skipUnless(HAS_NUMPY, "Numerical diagnostics require NumPy.")
    def test_parser_rejects_incomplete_extra_and_nonfinite_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.bin"
            header = struct.pack("<QQ", check.MAGIC, 4)
            block = struct.pack("<Q4f", 1, 1, 2, 3, 4)
            path.write_bytes(header + block)
            values, vocab = check.read_logits(path, 1)
            self.assertEqual(vocab, 4)
            self.assertEqual(values.tolist(), [[1, 2, 3, 4]])
            for data, count in ((header + block[:-1], 1), (header + block, 2),
                                (header + block + block, 1),
                                (header + struct.pack("<Q4f", 1, 1, 2, float("nan"), 4), 1)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    check.read_logits(path, count)

    @unittest.skipUnless(HAS_NUMPY, "Numerical diagnostics require NumPy.")
    def test_numeric_gate_rejects_an_altered_row(self):
        import numpy as np
        stock = np.ones((2, 4), dtype="float32")
        changed = stock.copy()
        changed[1, 2] += 0.1
        self.assertTrue(check.compare(stock, stock)["all_passed"])
        self.assertEqual(check.compare(stock, changed)["passed"], 1)

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++"),
                         "The transfer observer requires Linux and g++.")
    def test_interposer_preserves_transfer_and_observes_only_matching_successful_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / "copy.cpp"
            stub.write_text(
                '#include <cstring>\nextern "C" int cudaMemcpy(void* d,const void* s,size_t n,int k) '
                '{ if(k==99) return 7; std::memcpy(d,s,n); return 0; }\n')
            main = root / "main.cpp"
            main.write_text(
                '#include <cstddef>\nextern "C" int cudaMemcpy(void*,const void*,size_t,int);\n'
                'int main(){float a[4]={1,2,3,4},b[4]={}; '
                'if(cudaMemcpy(b,a,sizeof(a),99)!=7) return 1; '
                'if(cudaMemcpy(b,a,8,2)) return 2; '
                'if(cudaMemcpy(b,a,sizeof(a),1)) return 3; '
                'if(cudaMemcpy(b,a,sizeof(a),2)) return 4; '
                'for(int i=0;i<4;++i) if(a[i]!=b[i]) return 5; return 0;}\n')
            library = root / "capture.so"
            subprocess.run(["g++", "-shared", "-fPIC", "-std=c++17", "-pthread",
                            str(ROOT / "tools/trace/strata-logits-capture.cpp"), "-ldl", "-o", str(library)],
                           check=True, capture_output=True)
            subprocess.run(["g++", "-shared", "-fPIC", str(stub), "-o", str(root / "libcopy.so")],
                           check=True, capture_output=True)
            subprocess.run(["g++", str(main), "-L" + str(root), "-lcopy", "-o", str(root / "probe")],
                           check=True, capture_output=True)
            path = root / "rows.bin"
            env = {**os.environ, "LD_PRELOAD": str(library), "LD_LIBRARY_PATH": str(root),
                   "VRAMBRIDGE_STRATA_LOGITS": str(path), "VRAMBRIDGE_STRATA_VOCAB": "4"}
            subprocess.run([str(root / "probe")], env=env, check=True, capture_output=True)
            self.assertEqual(struct.unpack("<QQQ4f", path.read_bytes()), (check.MAGIC, 4, 1, 1, 2, 3, 4))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    @unittest.skipUnless(HAS_NUMPY, "Numerical diagnostics require NumPy.")
    def test_scored_alignment_omits_only_unscored_window_suffixes(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = np.array([[1, 2, 3, 4], [2, 3, 1, 4], [9, 8, 7, 6], [3, 1, 4, 2]], dtype="<f4")
            raw = root / "rows.bin"
            raw.write_bytes(struct.pack("<QQ", check.MAGIC, 4) +
                            b"".join(struct.pack("<Q", 1) + row.tobytes() for row in rows))
            lines = []
            for index, position in ((0, 10), (1, 11), (3, 10)):
                row = rows[index].astype("float64")
                top = int(np.argmax(row))
                lse = row[top] + np.log(np.exp(row - row[top]).sum())
                lines.append(f"{position}\t0\t{row[0]-lse:.9f}\t{top}\t{row[top]-lse:.9f}\t0\n")
            positions = root / "positions.tsv"
            positions.write_text("".join(lines))
            scored, info = check.scored_logits(raw, positions)
            self.assertEqual(scored.tolist(), rows[[0, 1, 3]].tolist())
            self.assertEqual(info["unscored_window_suffix_rows"], 1)
            rows[1, 0] += 0.1
            raw.write_bytes(struct.pack("<QQ", check.MAGIC, 4) +
                            b"".join(struct.pack("<Q", 1) + row.tobytes() for row in rows))
            with self.assertRaises(ValueError):
                check.scored_logits(raw, positions)

    def test_bridge_rejects_unconfirmed_mtp(self):
        class Engine:
            info = {"context": 4096, "kv": "fp16", "expert_slots": 3067,
                    "drafting": "off", "lookup": 0, "pcie_frac": "0.00"}
            closed = False

            def close(self):
                self.closed = True
        engine = Engine()
        with self.assertRaises(ValueError):
            bridge.verified_engine(engine, 4096, 3067)
        self.assertTrue(engine.closed)
        engine.info = {**engine.info, "drafting": "mtp"}
        self.assertIs(bridge.verified_engine(engine, 4096, 3067), engine)

    def test_bridge_help_does_not_read_model_configuration(self):
        env = dict(os.environ)
        env.pop("VRAMBRIDGE_STRATA_CONFIG", None)
        result = subprocess.run([sys.executable, str(ROOT / "tools/benchmark/vrambridge-strata-server.py"), "--help"],
                                env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Research-only", result.stdout)

    def test_native_completion_counts_accepted_tokens_and_releases_admission(self):
        class Tokenizer:
            def encode(self, text, parse_special=False):
                return [7, 8]

            def decode(self, tokens):
                return "answer"

        class Engine:
            last = {"generated": 3, "decode_ms": 6, "prompt_ms": 12, "drafts_offered": 5}

            def generate(self, *_):
                yield from (None, 1, 2, 3)

        class Service:
            tok, engine, stop_ids = Tokenizer(), Engine(), set()
            released = False

            def reported_ctx(self):
                return 1024

            def reserve_inference(self):
                return True

            def release_inference(self):
                self.released = True

            def _turn(self, _):
                return nullcontext()

        svc = Service()
        result = bridge.raw_completion(svc, {"prompt": "input", "n_predict": 3, "ignore_eos": True})
        self.assertEqual(result["timings"]["predicted_per_second"], 500)
        self.assertEqual(result["timings"]["predicted_n"], 3)
        self.assertTrue(svc.released)
        svc.released = False
        svc.engine.last = {**svc.engine.last, "generated": 4}
        with self.assertRaises(ValueError):
            bridge.raw_completion(svc, {"prompt": "input", "n_predict": 3, "ignore_eos": True})
        self.assertTrue(svc.released)


if __name__ == "__main__":
    unittest.main()
