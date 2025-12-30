import logging

from rich.logging import RichHandler


def setup_logging(verbosity: int = 0) -> None:
    if verbosity == 0:
        app_level, lib_level = logging.INFO, logging.WARNING
    elif verbosity == 1:
        app_level, lib_level = logging.DEBUG, logging.WARNING
    else:
        app_level, lib_level = logging.DEBUG, logging.DEBUG

    logging.basicConfig(
        level=lib_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=True,
            )
        ],
    )

    logging.getLogger("things_eeg2_dataset").setLevel(app_level)
