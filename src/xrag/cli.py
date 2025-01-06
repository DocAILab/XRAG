import argparse
import sys
from enum import Enum, unique
from .webui import run_web_ui
from .launcher import run
from .config import Config
from .data.qa_loader import generate_qa_from_folder
VERSION = "0.1.2"
USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   xrag-cli run -h: launch an eval experiment       |\n"
    + "|   xrag-cli webui: launch XRAGBoard                        |\n"
    + "|   xrag-cli version: show version info                      |\n"
    + "|   xrag-cli generate -i <input_file> -o <output_file> -n <num_questions> -s <sentence_length>: generate QA pairs from a folder |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + "| Welcome to XRAG, version {}".format(VERSION)
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/DocAILab/xrag |\n"
    + "-" * 58
)

@unique
class Command(str, Enum):
    RUN = "run"
    WEBUI = "webui"
    VER = "version"
    GENERATE = "generate"
    HELP = "help"

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='XRAG CLI Tool')

    # Define the subcommands
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    # 'run' command
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--override', nargs='*', help='Override config values (e.g., --override key1=value1 key2=value2)')

    # Other commands
    subparsers.add_parser('webui', help='Run the web UI')
    subparsers.add_parser('version', help='Show version')
    subparsers.add_parser('help', help='Show help')
    generate_parser = subparsers.add_parser('generate', help='Generate QA pairs')
    generate_parser.add_argument('-i', '--input', type=str, help='Input file path')
    generate_parser.add_argument('-o', '--output', type=str, help='Output file path')
    generate_parser.add_argument('-n', '--num', type=int, help='Number of questions per file', default=3)
    generate_parser.add_argument('-s', '--sentence_length', type=int, help='Sentence length, -1 means no split', default=-1)
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
    elif args.command == Command.GENERATE:
        generate_qa_from_folder(args.input, args.output, args.num, args.sentence_length)
    else:
        print(f"Unknown command: {args.command}")
