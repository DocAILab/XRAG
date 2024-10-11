import os
import random
import subprocess
import sys
from enum import Enum, unique
from .webui import run_web_ui
from .launcher import run
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
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.RUN:
        run()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
