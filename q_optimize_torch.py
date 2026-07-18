"""Compatibility entry point for the Torch latent-q backend."""

from lvs.backends.torch_mlp import *  # noqa: F401,F403
from lvs.backends.torch_mlp import main


if __name__ == "__main__":
    main()
