"""Compatibility entry point for the optional KAN latent-q backend."""

from lvs.backends.kan import *  # noqa: F401,F403
from lvs.backends.kan import main


if __name__ == "__main__":
    main()
