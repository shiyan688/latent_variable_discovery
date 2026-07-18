"""Allow running: python -m lvs <subcommand>"""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m lvs <subcommand>")
        print("  workflow      Run single-expression workflow")
        print("  batch         Run batch workflow")
        print("  expression-q  Run expression latent-q pipeline")
        print("  torch         Run the Torch latent-q backend on CSV files")
        sys.exit(1)

    commands = {
        "workflow": "lvs.workflows.single",
        "batch": "lvs.workflows.batch",
        "expression-q": "lvs.workflows.expression_latent_q",
        "torch": "lvs.backends.torch_mlp",
    }

    subcmd = sys.argv[1]
    if subcmd not in commands:
        print(f"Unknown subcommand: {subcmd}")
        print("Available: " + ", ".join(commands.keys()))
        sys.exit(1)

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module = __import__(commands[subcmd], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
