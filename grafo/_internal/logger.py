import logging

logger = logging.getLogger("grafo")
logger.setLevel(logging.WARNING)

formatter = logging.Formatter(
    "\033[92m%(levelname)s\033[0m\t(%(asctime)s) %(message)s", datefmt="%H:%M:%S"
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False
