#!/usr/bin/env python3
"""Reproduce the exact locally tested causal backend package without installing it."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


COMPONENTS = {
    "llama-server": ("62521cdb24abc16d25772e5c5b9c491bc554fe551c26a1386894502231986979",
                     "aaf14eb2a6ed697cbd4197126cfe1066cc49fa9971ac29a05db92455a7e94199"),
    "libllama-server-impl.so": ("4f06b44376952e6c3b7857cbf3cacee9f72dedc43a65e46e722b792ab23df863",
                                "1106099e3d8de07a46c2ccaf0e5c529d4283086bbbcc7b4f71761584b0b53863"),
    "libllama-common.so.0.7.0": ("b8476a596321362729a0345f25619a74346d01a703f6fee00f8ecfe8a6b6f2b1",
                                 "4e31151911c27be3a574be3c12e1a085914fb2b0edea8496074d0bec0f64d2f2"),
    "libllama.so.0.7.0": ("409dfaa4dd1f10c9471f804cc30de740388438d326cca10d92776c83025be845",
                          "c99600b55f68234b64846d85ba504c25e6cb9a0c3bc0419c12fb19103f40b1ec"),
    "libmtmd.so.0.7.0": ("b7bacc1fad0c02f7bbced3a1d4f043bc3f656ba43a54131a6820363793f4dc09",
                         "f8d0ac43e1fa00dae749a2dcb2caeaf7de406f43444310f3f0238f7c7f543f70"),
    "libggml.so.0.27.0": ("0dc5bad7cb9ea8e1cb9fe6fe5726bfffe56c8046e2ced8b78d8d789ea493174b",
                          "749a1e32564cd4a8c59d8e6c63502bd5cc6685366b62535d66db4b1c07da1382"),
    "libggml-base.so.0.27.0": ("c018ccaee7903a5eb82dadd7dc82bb35cec92cb179056e712adaae63694814ed",
                               "23bd5a4a49a34f53b568fdc9128886acfb0f40ead8897e9b1449cb7a9646a409"),
    "libggml-cpu.so.0.27.0": ("7f06357f58937a4579540a4cd1fd11a9c80c20e00a2fb015e6c458f9f248594e",
                              "6da5034d1923c243519e714ac03783598a96587f4ff2f7995332879bd51e2383"),
    "libggml-cuda.so.0.27.0": ("846f25212c08736bead9562031a09cfd6626025bafda64876bcfd8e3df9954cd",
                               "00a3e5a2c8a996a46122ed410bf78e454716a81a97fb366910688156eff0acda"),
}
LINKS = {
    "libllama-common.so.0": "libllama-common.so.0.7.0",
    "libllama.so.0": "libllama.so.0.7.0",
    "libmtmd.so.0": "libmtmd.so.0.7.0",
    "libggml.so.0": "libggml.so.0.27.0",
    "libggml-base.so.0": "libggml-base.so.0.27.0",
    "libggml-cpu.so.0": "libggml-cpu.so.0.27.0",
    "libggml-cuda.so.0": "libggml-cuda.so.0.27.0",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(source, corrected, output, patchelf):
    if output.exists():
        raise ValueError("The staging output must not already exist.")
    if subprocess.run([str(patchelf), "--version"], check=True, capture_output=True, text=True).stdout.strip() != "patchelf 0.19.3":
        raise ValueError("The exact measured patchelf version is required.")
    sources = {name: corrected if name == "libllama.so.0.7.0" else source / name for name in COMPONENTS}
    for name, path in sources.items():
        if not path.is_file() or sha256(path) != COMPONENTS[name][0]:
            raise ValueError(f"Unqualified source component: {name}.")
    output.mkdir(parents=True, mode=0o700)
    for name, path in sources.items():
        target = output / name
        shutil.copyfile(path, target)
        target.chmod(0o755)
        subprocess.run([str(patchelf), "--set-rpath", "$ORIGIN", str(target)], check=True)
        if sha256(target) != COMPONENTS[name][1]:
            raise ValueError(f"Relocated component differs from the measured build: {name}.")
    for link, target in LINKS.items():
        os.symlink(target, output / link)
        os.symlink(link, output / link.removesuffix(".0"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-bin", type=Path, required=True)
    parser.add_argument("--corrected-libllama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patchelf", type=Path, required=True)
    args = parser.parse_args()
    stage(args.prototype_bin, args.corrected_libllama, args.output, args.patchelf)
    print("Exact measured components staged; no installed profile was changed.")


if __name__ == "__main__":
    main()
