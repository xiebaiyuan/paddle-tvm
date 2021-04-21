# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile PaddlePaddle Models
===================
**Author**: `Xie Baiyuan <https://github.com/xiebaiyuan/>`_

This article is an introductory tutorial to deploy PaddlePaddle models with Relay.

For us to begin with, Paddle2ONNX and ONNX package must be installed.

A quick solution is to install protobuf compiler, and

.. code-block:: bash

    pip install onnx --user
    pip install paddle2onnx


or please refer to offical site.
https://github.com/onnx/onnx
https://github.com/PaddlePaddle/Paddle2ONNX


"""
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata



######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples! This model takes a single input image of size
# 224x224 and outputs a scaled image that is 3x greater than the input along each
# axis, a 672x672 image. Re-scale the cat image to fit this input shape then
# convert to `YCbCr`. The super resolution model will then be applied to the
# luminance (`Y`) channel.
from PIL import Image


start = 0
stop = 1 * 3 * 224 * 224
step = 1
dtype = 'float32'

ndarray = np.arange(start, stop, step, dtype)

img = ndarray.reshape((1, 3, 224, 224))

img.shape

target = "llvm"

input_name = "image"
shape_dict = {input_name: img.shape}


print(shape_dict)

######################################################################
# Compile the model with relay
# ---------------------------------------------
# Typically ONNX models mix model input values with parameter values, with
# the input having the name `1`. This model dependent, and you should check
# with the documentation for your model to determine the full input and
# parameter name space.
#
# Passing in the shape dictionary to the `relay.frontend.from_onnx` method
# tells relay which ONNX parameters are inputs, and which are parameters, and
# provides a static definition of the input size.
mobile_netv3_path="/workspace/tvm/tutorials/frontend/MobileNetV3_small_x1_0_not_combined"
mod, params = relay.frontend.from_paddlepaddle(mobile_netv3_path, shape_dict)
print("from_paddlepaddle end====>")


with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0),
                                               target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
print("Execute on TVM Started ... ")

tvm_output = intrp.evaluate()(tvm.nd.array(img.astype(dtype)),
                              **params).asnumpy()
print("Execute on TVM END ... ")

