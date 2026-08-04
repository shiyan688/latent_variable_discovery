#!/usr/bin/env python3
"""Run latent-q pipeline from expression library"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lvs.workflows.expression_latent_q import main

if __name__ == "__main__":
    main()
