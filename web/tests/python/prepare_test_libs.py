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
# Prepare test library for standalone wasm runtime test.

import tvm
from tvm import te
from tvm.contrib import emcc
from tvm import relay
import os

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

    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenetv2_1.0", pretrained=True)
    # block = get_model("mobilenetv2_0.5", pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
    net = mod["main"]
    net = relay.Function(
        net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    )
    mod = tvm.IRModule.from_expr(net)
    return mod, params, input_shape, output_shape, net

def prepare_test_libs(base_path):
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    fadd = tvm.build(s, [A, B], target, name="add_one")

    wasm_path = os.path.join(base_path, "test_addone.wasm")
    fadd.export_library(wasm_path, emcc.create_tvmjs_wasm)
    print("output: " + wasm_path)

def prepare_test_mobilenetlibs(base_path):
    network = "mobilenet"
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"  
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    model_name = "mxnet"
    batch_size = 1

    mod, params, data_shape, out_shape, func = get_network(model_name, batch_size)
    print("data_shape")
    print(data_shape)

    with tvm.transform.PassContext(opt_level=3):
        fadd = relay.build(mod, target, params=params, mod_name="mobilenet")
    # with relay.build_config(opt_level=3):
    #     graph, lib, params = relay.build(
    #         func, target=target, target_host=target, params=params)

    wasm_path = os.path.join(base_path, "mobilenet.wasm")

    lib.export_library(wasm_path, emcc.create_tvmjs_wasm)
    print("output: " + wasm_path)

    with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
        f_params.write(relay.save_param_dict(params))

if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "../../dist/wasm"))
    prepare_test_mobilenetlibs(os.path.join(curr_path, "../../dist/wasm"))
