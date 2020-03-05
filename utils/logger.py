import logging

LOG_LEVEL = logging.DEBUG
LOG_FILE = 'logger.log'
LOG_FILE_HANDLE = False

LOG_FORMAT = '%(asctime)s:%(levelname)s | %(filename)s:%(lineno)d-%(module)s | %(message)s'
LOG_FORMAT = logging.Formatter(LOG_FORMAT)


def set_logger(file_name):
  logger = logging.getLogger(file_name)
  logger.setLevel(LOG_LEVEL)

  if LOG_FILE_HANDLE:
    get_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    get_handler.setFormatter(LOG_FORMAT)
    logger.addHandler(get_handler)

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(LOG_FORMAT)
  logger.addHandler(stream_handler)

  return logger

if __name__ == "__main__":
  logger = set_logger(__name__)
  logger.debug("testtest")