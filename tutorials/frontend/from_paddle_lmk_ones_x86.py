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

For us to begin with, PaddlePaddle package must be installed.

.. code-block:: bash

   python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple  


or please refer to offical site.
https://www.paddlepaddle.org.cn/

"""
import os
import paddle
from paddle.static import load_inference_model 
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata

######################################################################
# Load PaddleDetection model
# ---------------------------------------------
# The example paddle detection model used here is exactly the same model in paddle 
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/static/configs/mobile/README.md
# we download the saved onnx model



model_path = '/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer'
print(model_path)

# now you have paddle detection on disk

paddle.enable_static()

[net_program, 
feed_target_names, 
fetch_targets] = paddle.static.load_inference_model(model_path, model_filename= 'model',params_filename='params',executor=paddle.static.Executor(paddle.fluid.CPUPlace()))



######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples! This model takes a single input image of size
# 224x224 and outputs a scaled image that is 3x greater than the input along each
# axis, a 672x672 image. Re-scale the cat image to fit this input shape then
# convert to `YCbCr`. The super resolution model will then be applied to the
# luminance (`Y`) channel.
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((64, 64))

img_arr = np.array(img)
print(img_arr.shape)


ones = np.ones([3,64,64])
image_chw = np.transpose(img_arr, (2,0,1))
print(image_chw.shape)

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
target = "llvm"

# target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"

shape_dict = {'image': [1,3,64,64]}
mod, params = relay.frontend.from_paddle(net_program, shape_dict)


with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(ones.astype(dtype)), **params)

print(tvm_output)

# print(tvm_output.shape)

# print(tvm_output.max())
# print(tvm_output.argmax())

