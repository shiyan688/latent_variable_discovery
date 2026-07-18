"""Compatibility entry point for batched benchmark workflows."""

from lvs.workflows.batch import *  # noqa: F401,F403
from lvs.workflows.batch import main


if __name__ == "__main__":
    main()
