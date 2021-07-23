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

# mod, params = relay.frontend.from_paddlepaddle(mobile_netv3_path)

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
mobile_netv3_path="/Users/dengyuguang/Downloads/MobileNetV2_SPLIT"
mod, params = relay.frontend.from_paddlepaddle(mobile_netv3_path, shape_dict)
print("from_paddlepaddle end====>")


with tvm.transform.PassContext(opt_level=3):
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
######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck. The luminance channel, `Y` is the output
# from the model. The chroma channels `Cb` and `Cr` are resized to match with a simple
# bicubic algorithm. The image is then recombined and converted back to `RGB`.
#from matplotlib import pyplot as plt

#out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
#out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
#out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
#result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
#canvas = np.full((672, 672 * 2, 3), 255)
#canvas[0:224, 0:224, :] = np.asarray(img)
#canvas[:, 672:, :] = np.asarray(result)
#plt.imshow(canvas.astype(np.uint8))
#plt.show()

######################################################################
# Notes
# ---------------------------------------------
# By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
# retains that dynamism upon import, and the compiler attemps to convert the model
# into a static shapes at compile time. If this fails, there may still be dynamic
# operations in the model. Not all TVM kernels currently support dynamic shapes,
# please file an issue on discuss.tvm.apache.org if you hit an error with dynamic kernels.
#
# This particular model was build using an older version of ONNX. During the import
# phase ONNX importer will run the ONNX verifier, which may throw a `Mismatched attribute type`
# warning. Because TVM supports a number of different ONNX versions, the Relay model
# will still be valid.
