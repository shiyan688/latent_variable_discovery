#!/usr/bin/env python3
"""Run single-expression latent-variable workflow"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lvs.workflows.single import main

if __name__ == "__main__":
    main()
