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

def func_save_to_file(func):
    wasm_path = "/Users/dengyuguang/opensource/tvm/web/dist/wasm/turning.wasm" 
    func.export_library(wasm_path, emcc.create_tvmjs_wasm)
    print("output: " + wasm_path)

def build_op_wasm():
    print("building")
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    fadd = tvm.build(s, [A, B], target, name="add_one")

    func_save_to_file(fadd)
    
dtype = "float32"

def build_op_gemm():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # Default schedule
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], target=target, name="mmult")

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)
# Blocking
def build_op_gemm_op1():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # schedule
    bn = 32
    s = te.create_schedule(C.op)

    # Blocking by loop tiling
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    # Hoist reduction domain outside the blocking loop
    s[C].reorder(xo, yo, ko, ki, xi, yi)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)

# Vectorization
def build_op_gemm_op2():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # Default schedule
    bn = 32
    s = te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    s[C].reorder(xo, yo, ko, ki, xi, yi)

    # Vectorization
    s[C].vectorize(yi)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)

# Loop Permutation
def build_op_gemm_op3():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # Default schedule
    bn = 32
    s = te.create_schedule(C.op)
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    # re-ordering
    s[C].reorder(xo, yo, ko, xi, ki, yi)
    s[C].vectorize(yi)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)


# Array Packing
def build_op_gemm_op4():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    bn = 32

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

    # Default schedule
    
    s = te.create_schedule(C.op)

    # Blocking by loop tiling
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (k,) = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)

    # Hoist reduction domain outside the blocking loop
    s[C].reorder(xo, yo, ko, ki, xi, yi)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)

# Write cache for blocks
def build_op_gemm_op5():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    bn = 32

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    # C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )

    # Default schedule
    
    s = te.create_schedule(C.op)

    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    # Write cache is computed at yo
    s[CC].compute_at(s[C], yo)

    # New inner axes
    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)

def build_op_gemm_op6():
    print("build gemm")

    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    M = 1024
    K = 1024
    N = 1024

    bn = 32

    # Algorithm
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    # C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
        name="C",
    )

    # Default schedule
    
    s = te.create_schedule(C.op)

    CC = s.cache_write(C, "global")

    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

    s[CC].compute_at(s[C], yo)

    xc, yc = s[CC].op.axis

    (k,) = s[CC].op.reduce_axis
    ko, ki = s[CC].split(k, factor=4)
    s[CC].reorder(ko, xc, ki, yc)
    s[CC].unroll(ki)
    s[CC].vectorize(yc)

    # parallel
    s[C].parallel(xo)

    x, y, z = s[packedB].op.axis
    s[packedB].vectorize(z)
    s[packedB].parallel(x)

    func = tvm.build(s, [A, B, C], target=target, name="mmult")
    assert func

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    func_save_to_file(func)

def run_rpctracker():
    proxy_host = "0.0.0.0"
    proxy_port = 9192

    if not tvm.runtime.enabled("rpc"):
        return
    # generate the wasm library
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)
    
    model_json_path = "/Users/dengyuguang/opensource/tvm/web/tests/python/../../dist/wasm/mobilenet.json" 
    model_param_path = "/Users/dengyuguang/opensource/tvm/web/tests/python/../../dist/wasm/mobilenet.params" 

    model_json = open(model_json_path, 'r').read()

    print("loading wasm file")

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

    remote = tracker.request("wasm", priority=0, session_timeout=60, session_constructor_args=["rpc.WasmSession", 1])
    def run(remote):
        ctx = remote.cpu(0)
        M = 1024
        K = 1024
        N = 1024

        # Random generated tensor for testing
        a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
        b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
        
        f1 = remote.system_lib()
        func = f1.get_function("mmult")

        c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)

        # print("run once")
        # func(a, b, c)
        # answer = np.dot(a.asnumpy(), b.asnumpy())
        # tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
        print("check result")

        evaluator = f1.time_evaluator("mmult", ctx, number=5)
        print("Baseline: %f" % evaluator(a, b, c).mean)

    run(remote)

# python -m tvm.exec.rpc_proxy --example-rpc=1
# test_predict()

# python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
# python -m tvm.exec.rpc_proxy --example-rpc=1 --tracker=0.0.0.0:9190
# python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=client:wasm

# build_op_gemm() #Baseline: 3.962023
# build_op_gemm_op1() # Baseline: 2.713712
# build_op_gemm_op2() # Baseline: 1.178124
# build_op_gemm_op3() # Baseline: 0.671167
# build_op_gemm_op4() # Baseline: 0.528168
# build_op_gemm_op5() # Baseline: 0.427384
# build_op_gemm_op6() # Baseline: 0.432338 不支持 thread-level parallelization
run_rpctracker()



