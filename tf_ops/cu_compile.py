import os
import sys

import tensorflow as tf
import subprocess


BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

from utils.logger import set_logger
import utils.config_ini as config

logger = set_logger(__name__)


"""
Operation作成に伴うcuda, cppのコンパイル
https://www.tensorflow.org/guide/create_op
"""
# compileオプション
TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())

# リンクディレクトリ
TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())

GROUPING_PATH = BASE_DIR + '/tf_ops/grouping'
INTERPOLATION_PATH = BASE_DIR + '/tf_ops/interpolation'
SAMPLING_PATH = BASE_DIR + '/tf_ops/sampling'

NVCC_CMD = os.path.join(config.CUDA_PATH, 'bin', 'nvcc')
CUDA_CFLAG = '-I ' + os.path.join(config.CUDA_PATH, 'include')
CUDA_LFLAG = '-lcudart -L ' + os.path.join(config.CUDA_PATH, 'lib64') + '/'


def confirm_dir(path):
    if os.path.isdir(path):
        return
    raise Exception("{} がありません".format(path))

def confirm_file(path):
    if os.path.isfile(path):
        return
    raise Exception("{} がありません".format(path))

confirm_dir(GROUPING_PATH)
confirm_dir(INTERPOLATION_PATH)
confirm_dir(SAMPLING_PATH)
confirm_file(config.CUDA_PATH)


def nvcc_grouping():
    return NVCC_CMD + ' ' + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu') + ' -o ' \
           + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu.o') \
           + ' -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'


def nvcc_sampling():
    return NVCC_CMD + ' ' + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu') + ' -o ' \
           + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu.o') \
           + ' -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'


def cpp_sampling():
    return 'g++ -std=c++11 ' + os.path.join(SAMPLING_PATH, 'tf_sampling.cpp') \
           + ' ' + os.path.join(SAMPLING_PATH, 'tf_sampling_g.cu.o') + ' -o ' \
           + os.path.join(SAMPLING_PATH, 'tf_sampling_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + CUDA_CFLAG + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'

def cpp_interpolation():
    return 'g++ -std=c++11 ' + os.path.join(INTERPOLATION_PATH, 'tf_interpolate.cpp') + ' -o ' \
           + os.path.join(INTERPOLATION_PATH, 'tf_interpolate_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + CUDA_CFLAG + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'

def cpp_grouping():
    return 'g++ -std=c++11 ' + os.path.join(GROUPING_PATH, 'tf_grouping.cpp') \
           + ' ' + os.path.join(GROUPING_PATH, 'tf_grouping_g.cu.o') + ' -o ' \
           + os.path.join(GROUPING_PATH, 'tf_grouping_so.so') + ' -shared -fPIC ' \
           + TF_CFLAGS + ' ' + CUDA_CFLAG + ' ' + TF_LFLAGS + ' ' + CUDA_LFLAG + ' -O2'


def nvcc_proc_call(gcolab = False):
    gr = nvcc_grouping()
    print("!" + gr)
    if gcolab is False:
        try:
            res = subprocess.check_call(gr)
            logger.debug(res)
        except Exception as e:
            logger.error(e)

    sm = nvcc_sampling()
    print("!" + sm)
    if gcolab is False:
        try:
            res = subprocess.check_call(sm)
            logger.debug(res)
        except Exception as e:
            logger.error(e)


def cpp_proc_call(gcolab = False):
    gr = cpp_grouping()
    print("!" + gr)
    if gcolab is False:
        try:
            res = subprocess.check_call(gr)
            logger.debug(res)
        except Exception as e:
            logger.error(e)

    sm = cpp_sampling()
    print("!" + sm)
    if gcolab is False:
        try:
            res = subprocess.check_call(sm)
            logger.debug(res)
        except Exception as e:
            logger.error(e)

    inter = cpp_interpolation()
    print("!" + inter)
    if gcolab is False:
        try:
            res = subprocess.check_call(inter)
            logger.debug(res)
        except Exception as e:
            logger.error(e)

if __name__ == "__main__":
    nvcc_proc_call()
    cpp_proc_call()