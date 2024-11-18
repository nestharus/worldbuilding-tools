import argparse
import logging
import sys

from rich.console import Console

from model_installer import ModelInstaller
from setup_logging import setup_rich_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Model setup and installation tool")
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Force update of models even if they already exist"
    )
    return parser.parse_args()


def main():
    setup_rich_logging()
    logger = logging.getLogger(__name__)

    console = Console()

    with console.status("[bold blue]Starting setup...") as status:
        try:
            installer = ModelInstaller()

            status.update("[bold yellow]Installing models...")
            installer.install_all()

            console.print("[bold green]Setup completed successfully![/]")

        except Exception as e:
            console.print(f"[bold red]Setup failed: {e}[/]")
            logger.exception("Setup failed with exception:")
            return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
