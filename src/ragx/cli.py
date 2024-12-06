import argparse
import sys
from enum import Enum, unique
from .webui import run_web_ui
from .launcher import run
from .config import Config
VERSION = "0.1.0"
USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   ragx-cli run -h: launch an eval experiment       |\n"
    + "|   ragx-cli webui: launch RAGXBoard                        |\n"
    + "|   ragx-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + "| Welcome to RAGX, version {}".format(VERSION)
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/DocAILab/RAGX |\n"
    + "-" * 58
)

@unique
class Command(str, Enum):
    RUN = "run"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='RAGX CLI Tool')

    # Define the subcommands
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # 'run' command
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--override', nargs='*', help='Override config values (e.g., --override key1=value1 key2=value2)')

    # Other commands
    subparsers.add_parser('webui', help='Run the web UI')
    subparsers.add_parser('version', help='Show version')
    subparsers.add_parser('help', help='Show help')

    # Parse the arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == Command.RUN:
        # Parse overrides
        config_overrides = {}
        if args.override:
            for override in args.override:
                if '=' in override:
                    key, value = override.split('=', 1)
                    config_overrides[key.strip()] = value.strip()
                else:
                    print(f"Invalid override format: {override}")
                    sys.exit(1)
        # Update the Config instance
        config = Config()
        config.update_config(config_overrides)
        run()
    elif args.command == Command.WEBUI:
        run_web_ui()
    elif args.command == Command.VER:
        print(WELCOME)
    elif args.command == Command.HELP or args.command is None:
        parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
