import datetime
from enum import Enum
from colorama import Fore, Back, Style, init

init()


class LogLevel(Enum):
    NEEDIMPL = 0,
    TODO = 1,
    DEBUG = 2,
    INFO = 3,
    WARNING = 4,
    ERROR = 5,
    CRITICAL = 6


selected_log_level = LogLevel.DEBUG

colorMappings = {
    LogLevel.NEEDIMPL: Fore.WHITE + Back.BLACK,
    LogLevel.TODO: Fore.GREEN,
    LogLevel.DEBUG: Fore.BLUE,
    LogLevel.INFO: Fore.BLACK,
    LogLevel.WARNING: Fore.YELLOW,
    LogLevel.ERROR: Fore.LIGHTRED_EX,
    LogLevel.CRITICAL: Fore.BLACK + Back.RED
}


def select_log_level(level: LogLevel):
    global selected_log_level
    selected_log_level = level


def line(char, length, msg):
    print("{:{}^{}}".format(msg, char, length))


def full_line(char="*", msg=""):
    line(char, 64, msg)


def prnt(level, msg, new_line):
    color = colorMappings.get(level)
    dt = datetime.datetime.now()
    print("{}[{:>8} {:%H:%M:%S}] {}{}".format(color, level.name, dt, msg, Style.RESET_ALL),
          end='\n' if new_line else "")


def todo(msg, new_line=True):
    if selected_log_level.value <= LogLevel.TODO.value:
        prnt(LogLevel.TODO, msg, new_line)


def debug(msg, new_line=True):
    if selected_log_level.value <= LogLevel.DEBUG.value:
        prnt(LogLevel.DEBUG, msg, new_line)


def info(msg, new_line=True):
    if selected_log_level.value <= LogLevel.INFO.value:
        prnt(LogLevel.INFO, msg, new_line)


def warn(msg, new_line=True):
    if selected_log_level.value <= LogLevel.WARNING.value:
        prnt(LogLevel.WARNING, msg, new_line)


def need_impl(msg, new_line=True):
    if selected_log_level.value <= LogLevel.TODO.value:
        full_line("-", msg="Missing implementation")
        prnt(LogLevel.NEEDIMPL, msg, new_line)
        full_line("-")


def error(msg, new_line=True):
    prnt(LogLevel.ERROR, msg, new_line)


def critical(msg, new_line=True):
    prnt(LogLevel.CRITICAL, msg, new_line)
