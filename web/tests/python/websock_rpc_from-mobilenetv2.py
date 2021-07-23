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
"""Simple testcode to test Javascript RPC

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
Connect javascript end to the websocket port and connect to the RPC.
"""

import tvm
from tvm import te
from tvm import rpc
from tvm import relay
from tvm.contrib import utils, emcc
import numpy as np

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can either load some pre-defined network from :code:`relay.testing`
# or building :any:`relay.testing.resnet` with relay.
# We can also load models from MXNet, ONNX and TensorFlow.
#
# In this tutorial, we choose resnet-18 as tuning example.


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "data"
    dtype = "float32"

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("mobilenetv2_1.0", pretrained=True)
        # block = get_model("mobilenetv2_0.5", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def test_predict():
    proxy_host = "127.0.0.1"
    proxy_port = 9090

    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    
    wasm_path = "/Users/dengyuguang/opensource/tvm/web/tests/python/../../dist/wasm/turning.wasm"
    model_json_path = "/Users/dengyuguang/opensource/tvm/web/tests/python/../../dist/wasm/mobilenet.json" 
    model_param_path = "/Users/dengyuguang/opensource/tvm/web/tests/python/../../dist/wasm/mobilenet.params" 

    model_json = open(model_json_path, 'r').read()

    print("loading wasm file")

    wasm_binary = open(wasm_path, "rb").read()

    print("connecting")
    
    remote = rpc.connect(
        proxy_host,
        proxy_port,
        key="wasm",
        session_constructor_args=["rpc.WasmSession", 1]
    )

    print("connected")
    print(remote)

    def check(remote):
        print("in check")
        # basic function checks.
        faddone = remote.get_function("testing.asyncAddOne")
        print(faddone(100))
        
        print("find cpu ctx")
        ctx = remote.cpu(0)
        # print(ctx.__dict__)
        # print(dir(ctx))

        # print(ctx.device_type)
        # print(ctx.device_id)

        # return;

        systemModuleLib = remote.system_lib()
        # print("call remote system lib")
        # print(systemModuleLib)
        # print(dir(systemModuleLib))

        fcreate = remote.get_function("tvm.graph_runtime.create")
        print(dir(ctx))
        executor = fcreate(model_json, systemModuleLib,1, ctx.device_id) # 1 = cpu type

        floadParams = executor.get_function("load_params")
        fsetInput = executor.get_function("set_input")
        fgetOutput = executor.get_function("get_output")
        frun = executor.get_function("run")

        outputTensor = fgetOutput(0)

        with open(model_param_path, "rb") as f:
            data = f.read()
            floadParams(data)

        frun()

        data_shape = (1, 3, 224, 224)
        x = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"), ctx)
        print(x.shape)
        fsetInput("data", x)
        frun()
        # print(fgetOutput(0))

        time_f =  executor.time_evaluator("run", ctx, repeat=1)
        cost = time_f().mean
        print(cost)

    check(remote)

def test_rpctracker():
    proxy_host = "0.0.0.0"
    proxy_port = 9190

    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    
    turning_path = "/Users/dengyuguang/opensource/tvm/web/apps/mobilenet/dist/mobilenet.wasm"
    model_json_path = "/Users/dengyuguang/opensource/tvm/web/apps/mobilenet/dist/mobilenet.json" 
    model_param_path = "/Users/dengyuguang/opensource/tvm/web/apps/mobilenet/dist/mobilenet.params" 

    model_json = open(model_json_path, 'r').read()

    print("loading wasm file")

    wasm_binary = open(turning_path, "rb").read()

    print("connecting")
    tracker = rpc.connect_tracker(
        proxy_host,
        proxy_port,
    )

    print("connected")

    print(tracker.text_summary())
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
        fcreate = remote.get_function("tvm.graph_runtime.create")
        # print(dir(ctx))

        print("load model")

        executor = fcreate(model_json, systemModuleLib,1, ctx.device_id) # 1 = cpu type

        floadParams = executor.get_function("load_params")
        fsetInput = executor.get_function("set_input")
        fgetOutput = executor.get_function("get_output")
        frun = executor.get_function("run")

        outputTensor = fgetOutput(0)

        with open(model_param_path, "rb") as f:
            data = f.read()
            floadParams(data)

        data_shape = (1, 3, 224, 224)
        x = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"), ctx)
        # print(x.shape)
        fsetInput("data", x)
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



