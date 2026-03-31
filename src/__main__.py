"""
main
"""

from pathlib import Path
from argparse import ArgumentParser, Namespace
from settings.config_loader import load_config_from_file
import logging
import importlib
from dotenv import load_dotenv
import os
import sys

sys.path.append(str(Path(__file__).resolve().parent))
# as we have not yet decided to make the project a package.

logger = logging.getLogger(Path(__file__).stem)


def parse_args() -> Namespace:
    """Parse arguments from CLI

    Returns:
        namespace containing parsed arguments in its attributes
    """

    parser = ArgumentParser(description="Parse arguments from CLI")
    parser.add_argument("config", nargs="+", help="Config files for a job")

    return parser.parse_args()


def load_config(list_config_file):

    CONF_DIR = Path(__file__).resolve().parent.parent / "config" / os.getenv("ENV")

    if not CONF_DIR.exists():
        raise FileNotFoundError(f"Config directory not found: {CONF_DIR}")

    path = CONF_DIR / list_config_file[0]

    return load_config_from_file(path)


def main() -> None:
    """Run the desired application"""

    load_dotenv()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    args = parse_args()
    list_config_file = args.config

    job_config = load_config(list_config_file)

    job_name = Path(list_config_file[-1]).stem

    job_module = importlib.import_module(name=f"application.{job_name}")

    logger.info(f"Launching job {job_name}...")

    job_module.job(config=job_config)


if __name__ == "__main__":
    main()
