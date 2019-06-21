# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""Import core names of TensorFlow.

Programs that want to build Brain Ops and Graphs without having to import the
constructors and utilities individually can import this file:

import tensorflow.python.platform
import tensorflow as tf

"""

import tensorflow.python.platform
# [R]以下的_pb2是由.proto 编译自动生成的python 代码
from tensorflow.core.framework.graph_pb2 import *
from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Framework
# [R] 框架，定义整个计算图相关的类。如 Graph/Tensor等
from tensorflow.python.framework.framework_lib import *

# Session
# [R] 发起图计算相关。主要为 Session
from tensorflow.python.client.client_lib import *

# Ops
# [R]所有的操作，且分门别类，有数值计算类 OP，随机 OP，IO OP，控制流 Op 等等
from tensorflow.python.ops.standard_ops import *

# Bring nn, image_ops, user_ops as a subpackages
from tensorflow.python.ops import nn
from tensorflow.python.ops import image_ops as image
from tensorflow.python.user_ops import user_ops

# Import the names from python/training.py as train.Name.
# [R] 训练相关的工具，比如，优化器，协调器、QueueRunner 等
from tensorflow.python.training import training as train

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
# [R] 系统相关的环境，如日志、测试、命令行参数等等
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
from tensorflow.python.platform import test
