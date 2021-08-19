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
import paddle.fluid as fluid
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import tarfile

######################################################################
# Load PaddleDetection model
# ---------------------------------------------
# The example paddle detection model used here is exactly the same model in paddle 
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/static/configs/mobile/README.md
# we download the saved onnx model
model_local = "/data/lmk_demo/lmk_demo"


# download_model_path = download_testdata(model_url_mbv1, "MobileNetV1.tar.gz", module="paddle")
# import os  

# def un_targz(file_name,dst_path):  
#     import subprocess
#     command = 'tar -zxvf {} -C {}'.format(file_name, dst_path)
#     subprocess.call(command, shell=True)
# model_parent = os.path.dirname(download_model_path)
# print(model_parent)
# un_targz(download_model_path,model_parent)

# model_path = model_parent + '/MobileNetV1'
# print(model_path)

# now you have paddle detection on disk

paddle.enable_static()
def load_inference_model(model_path, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, '__model__')
    param_abs_path = os.path.join(model_path, '__params__')
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, '__model__', '__params__')
    else:
        return fluid.io.load_inference_model(model_path, exe)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
scope = fluid.core.Scope()

[net_program, 
feed_target_names, 
fetch_targets] = load_inference_model(model_local, exe)
# print(net_program)
global_block = net_program.global_block()





    # feed_list = feed_ones(global_block, feed_target_names, 1)
    # #feed_list = feed_randn(global_block, feed_target_names, 1, need_save=True)
    # fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
    # results = exe.run(program=net_program,
    #                     feed=feed_list,
    #                     fetch_list=fetch_targets,
    #                     return_numpy=False)
    # print ("123")
    # #for var_ in net_program.list_vars():
    # #  print var_
    # #print list(filter(None, net_program.list_vars()))
    # fluid.io.save_params(executor=exe, dirname="./123", main_program=net_program)
    # print_results(results, fetch_targets, need_save=False)





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
img = Image.open(img_path).resize((224, 224))
img_ycbcr = img.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img_y)[np.newaxis, np.newaxis, :, :]

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

input_name = "1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_paddle(net_program, shape_dict)

with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).numpy()

######################################################################
# Display results
# ---------------------------------------------
# We put input and output image neck to neck. The luminance channel, `Y` is the output
# from the model. The chroma channels `Cb` and `Cr` are resized to match with a simple
# bicubic algorithm. The image is then recombined and converted back to `RGB`.
from matplotlib import pyplot as plt

out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)), mode="L")
out_cb = img_cb.resize(out_y.size, Image.BICUBIC)
out_cr = img_cr.resize(out_y.size, Image.BICUBIC)
result = Image.merge("YCbCr", [out_y, out_cb, out_cr]).convert("RGB")
canvas = np.full((672, 672 * 2, 3), 255)
canvas[0:224, 0:224, :] = np.asarray(img)
canvas[:, 672:, :] = np.asarray(result)
plt.imshow(canvas.astype(np.uint8))
plt.show()

######################################################################
# Notes
# ---------------------------------------------
# By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
# retains that dynamism upon import, and the compiler attempts to convert the model
# into a static shapes at compile time. If this fails, there may still be dynamic
# operations in the model. Not all TVM kernels currently support dynamic shapes,
# please file an issue on discuss.tvm.apache.org if you hit an error with dynamic kernels.
#
# This particular model was build using an older version of ONNX. During the import
# phase ONNX importer will run the ONNX verifier, which may throw a `Mismatched attribute type`
# warning. Because TVM supports a number of different ONNX versions, the Relay model
# will still be valid.
