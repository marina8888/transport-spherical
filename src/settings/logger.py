import logging


class LogConfig:
    @staticmethod
    def configure_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

        return logger