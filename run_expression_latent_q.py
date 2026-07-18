"""Compatibility entry point for expression latent-q experiments."""

from lvs.workflows.expression_latent_q import *  # noqa: F401,F403
from lvs.workflows.expression_latent_q import main


if __name__ == "__main__":
    main()
