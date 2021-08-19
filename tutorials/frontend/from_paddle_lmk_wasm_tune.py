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
from tvm.contrib import utils, emcc


from tvm import rpc
from tvm import relay
from tvm.contrib import utils, emcc



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



def test_rpctracker():
    proxy_host = "0.0.0.0"
    proxy_port = 9190

    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    
    turning_path = "/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/tune/lmk.wasm"
    model_json_path = "/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/tune/lmk.json" 
    model_param_path = "/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/tune/lmk.params" 

    model_json = open(model_json_path, 'r').read()

    print("loading wasm file")

    wasm_binary = open(turning_path, "rb").read()

    print("connecting")
    tracker = rpc.connect_tracker(
        proxy_host,
        proxy_port,
    )

    print("connected")

    # 这里会爆出来exception的问题. 
    # print(tracker.text_summary())

    remote = tracker.request("wasm", priority=0, session_timeout=60, session_constructor_args=["rpc.WasmSession", 1])

    print("requested")

    print(remote._sess)
    print(dir(remote))

    def check_tracker(remote):
        print("in check")
        # print("find cpu ctx")
        ctx = remote.cpu(0)

        systemModuleLib = remote.system_lib()
        
        # basic function checks.
        # fupload = remote.get_function("tvm.rpc.server.upload")
        # print(fupload("filepath", 1))
        # print("tvm.rpc.server.upload")

        fremove = remote.get_function("tvm.rpc.server.remove")
        fremove("filepath")

        systemModuleLib = remote.system_lib()
        fcreate = remote.get_function("tvm.graph_executor.create")
        # print(dir(ctx))

        print("load model")
        print(ctx)
        print("before fcreate : "+str(ctx.device_id))
        # executor = fcreate(model_json, systemModuleLib,1, ctx.device_id) # 1 = cpu type
        executor = fcreate(model_json, systemModuleLib,1 ,ctx.device_id) # 1 = cpu type

        floadParams = executor.get_function("load_params")
        fsetInput = executor.get_function("set_input")
        fgetOutput = executor.get_function("get_output")
        frun = executor.get_function("run")

        outputTensor = fgetOutput(0)

        with open(model_param_path, "rb") as f:
            data = f.read()
            floadParams(data)

        data_shape = (1, 3, 64, 64)
        x = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"), ctx)
        # print(x.shape)
        fsetInput("image", x)
        # frun()

        print("before run")
        time_f =  executor.time_evaluator("run", ctx, repeat=5)
        res = time_f()
        print(res.mean)
        print(res.results)

        # print(fgetOutput(0))

    def check_tracker_module(remote):
        print("in check")
        # print("find cpu ctx")
        ctx = remote.cpu(0)
        
        # # basic function checks.
        # fupload = remote.get_function("tvm.rpc.server.upload")
        # print(fupload("filepath", 1))
        # print("tvm.rpc.server.upload")

        # fremove = remote.get_function("tvm.rpc.server.remove")
        # fremove("filepath")

        print("load_module")
        systemModuleLib = remote.system_lib()

        print(systemModuleLib.entry_name)

        # frun = systemModuleLib.get_function("default_function")
        # frun(0,0,0);

# (((1, 2, 112, 112, 16), 'float32'), ((2, 3, 3, 3, 1, 16), 'float32'), ((1, 3, 224, 224, 1), 'float32'))

# (((1, 4, 112, 112, 8), 'float32'), ((4, 1, 3, 3, 3, 8), 'float32'), ((1, 1, 224, 224, 3), 'float32')),

        data_type = "float32"
        print("prepare call in remote")

        time_f = systemModuleLib.time_evaluator(
                "default_function",
                ctx,
                number=2,
                repeat=4,
        )

        arg_info = [(1, 2, 112, 112, 16), (2, 1, 3, 3, 3, 16), (1, 1, 224, 224, 3)]
        args = [tvm.nd.array(np.ones(x, dtype=data_type), ctx=ctx) for x in arg_info]

        for i in args:
            print(i.shape)

        ctx.sync()
        cost = time_f(*args).results
        print(cost)

        print("prepare call in remote end")

    # check_tracker_module(remote)
    check_tracker(remote)

# python -m tvm.exec.rpc_proxy --example-rpc=1
# test_predict()

# python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
# python -m tvm.exec.rpc_proxy --example-rpc=1 --tracker=0.0.0.0:9190
# python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=client:wasm
test_rpctracker()







exit(0)

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
# target = "llvm"

# target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"

shape_dict = {'image': [1,3,64,64]}
mod, params = relay.frontend.from_paddle(net_program, shape_dict)

network = 'lmk'
def prepare_test_mobilenetlibs(mod,params,network,base_path):
    '''Prepare Wasm Libs to deploy'''
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    batch_size = 1
    
    wasm_path = os.path.join(base_path, f"{network}.wasm")
    
    # relay 打包runtiem，模型参数，模型结构3个文件
    with tvm.transform.PassContext(opt_level=3):
        # graph, lib, params = relay.build(mod, target, params=params)
        executor_config, mod, params = relay.build(mod, target, params=params)
        print(executor_config)

    mod.export_library(wasm_path, emcc.create_tvmjs_wasm)
    print("output: " + wasm_path)

    with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
        f_graph_json.write(executor_config)
    with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
        f_params.write(relay.save_param_dict(params))

    
prepare_test_mobilenetlibs(mod=mod,params=params,network='lmk',base_path=model_path)



exit(1)

with tvm.transform.PassContext(opt_level=1):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = intrp.evaluate()(tvm.nd.array(image_chw.astype(dtype)), **params)

print(tvm_output)

# print(tvm_output.shape)

# print(tvm_output.max())
# print(tvm_output.argmax())

