"""Compatibility entry point for a single end-to-end workflow."""

from lvs.workflows.single import *  # noqa: F401,F403
from lvs.workflows.single import main


if __name__ == "__main__":
    main()
