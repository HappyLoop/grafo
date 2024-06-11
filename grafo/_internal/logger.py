import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("grafo")

formatter = logging.Formatter(
    "\033[92m%(levelname)s\033[0m\t(%(asctime)s) %(message)s", datefmt="%H:%M:%S"
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logging.getLogger("instructor").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
