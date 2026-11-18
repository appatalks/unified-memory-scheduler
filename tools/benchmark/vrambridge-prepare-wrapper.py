#!/usr/bin/env python3
"""Opt into existing CPU preparation reuse in a copied llama-server runtime."""

import os
from pathlib import Path
import sys


def main():
    server = str(Path(__file__).resolve().with_name('llama-server'))
    os.environ['VRAMBRIDGE_HYBRID_CPU_PREPARE'] = '1'
    os.execv(server, [server, *sys.argv[1:]])


if __name__ == '__main__':
    main()
