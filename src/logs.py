"""Console output of a script, also appended to a log file."""
import sys
import traceback
import warnings
from datetime import datetime


class Tee:
    """Writes to a console stream and to a file."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.file.write(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ... of the console stream
        return getattr(self.stream, name)


def keep_log(path):
    """From now on stdout, warnings and an uncaught exception of this process are also appended to path, after a line
    with the command. Progress bars (stderr) stay on the console."""
    log = open(path, "a", buffering=1)
    log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)}\n")
    sys.stdout = Tee(sys.stdout, log)
    show, hook = warnings.showwarning, sys.excepthook

    def showwarning(message, category, filename, lineno, file=None, line=None):
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
        show(message, category, filename, lineno, file, line)

    def excepthook(*exc_info):
        log.write("".join(traceback.format_exception(*exc_info)))
        hook(*exc_info)

    warnings.showwarning, sys.excepthook = showwarning, excepthook
