import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

import configparser

from utils.logger import set_logger

logger = set_logger(__name__)



_config = configparser.ConfigParser()
_config.sections()
_config.read(BASE_DIR + '/config.ini')

print(_config)
if 'ENV' in _config:
    CUDA_PATH = _config['ENV']['CUDA_PATH']
    if CUDA_PATH == "":
        raise Exception("CUDAのパスが設定されていません")

else:
    raise Exception("設定ファイルが間違っています")

logger.debug("CUDA_PATH: {}".format(CUDA_PATH))

logger.info("設定ファイルの読み込みが完了しました。")