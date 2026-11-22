"""Invoke the native CLI with identical initial RNG state for each paired arm."""

import runpy
import sys


def main():
    from pytorch_lightning import seed_everything

    seed_everything(int(sys.argv.pop(1)), workers=True)
    runpy.run_module("promera", run_name="__main__")


if __name__ == "__main__":
    main()
