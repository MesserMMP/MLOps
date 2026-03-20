import logging


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("ml-service")
