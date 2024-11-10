import argparse
import logging
import sys

from rich.console import Console

from cleanup import ModelCleaner
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
    args = parse_args()
    setup_rich_logging()
    logger = logging.getLogger(__name__)

    console = Console()

    with console.status("[bold blue]Starting setup...") as status:
        try:
            # Initialize installer
            installer = ModelInstaller()

            # Parse arguments
            parser = argparse.ArgumentParser(description='Setup and update tokenizer models')
            parser.add_argument('--force-update', action='store_true',
                                help='Force update all models even if up to date')
            args = parser.parse_args()

            # System check with progress
            status.update("[bold yellow]Checking system resources...")
            resources = installer.sys_check.check_system()
            logger.info(f"System resources: {resources}")

            # Installation process
            status.update("[bold yellow]Installing models...")
            installer.install_all(force_update=args.force_update)

            # Final verification
            status.update("[bold yellow]Performing final verification...")
            cleaner = ModelCleaner()
            cleaner.verify_models()

            console.print("[bold green]Setup completed successfully![/]")

        except Exception as e:
            console.print(f"[bold red]Setup failed: {e}[/]")
            logger.exception("Setup failed with exception:")
            return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
