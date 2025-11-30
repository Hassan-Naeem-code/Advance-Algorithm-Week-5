"""Small CLI for running the pipeline and experiments.
Usage: `python -m src.cli run` or `python -m src.cli experiments`
"""
import argparse
import logging
from src.logging_config import configure_logging


def _build_parser():
    p = argparse.ArgumentParser(prog="assignment05")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the main pipeline (train/evaluate)")
    run.add_argument("--data", help="Path to CSV (optional)", default=None)

    exp = sub.add_parser("experiments", help="Run experiments suite")
    exp.add_argument("--data", help="Path to CSV (optional)", default=None)

    return p


def main(argv=None):
    configure_logging()
    logger = logging.getLogger(__name__)
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        from src.main import main as run_main
        logger.info("Running main pipeline")
        run_main()
    elif args.cmd == "experiments":
        from src.experiments import main as run_experiments
        logger.info("Running experiments suite")
        run_experiments()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
