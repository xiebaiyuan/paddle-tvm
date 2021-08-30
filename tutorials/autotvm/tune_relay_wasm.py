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
.. _tune_relay_wasm:

Auto-tuning a Convolutional Network for Wasm
===============================================
**Author**: `DengYuguang <dengyuguang@baidu.com>`_,`XieBaiyuan <https://github.com/xiebaiyuan>`

Auto-tuning for a specific Wasm Js is critical for getting the best
performance. This is a tutorial about how to tune a whole convolutional
network.

The operator implementation for Js/Wasm in TVM is written in template form.
The template has many tunable knobs (tile factor, vectorization, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os

import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_executor as runtime
from tvm.contrib import emcc

import wasm_module_loader

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

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
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size)
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

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses RPC session to communicate with ARM boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up the tuning, TVM uses RPC Tracker to manage distributed devices.
# The RPC Tracker is a centralized controller node. We can register all devices to
# the tracker. For example, if we have 10 phones, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register Devices to RPC Tracker
# -----------------------------------
# Now we can register our devices to the tracker. The first step is to
# build the TVM runtime for the ARM devices.
#
# * For Linux:
#   Follow this section :ref:`build-tvm-runtime-on-device` to build
#   the TVM runtime on the device. Then register the device to tracker by
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rk3399
#
#   (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# * For Android:
#   Follow this `readme page <https://github.com/apache/tvm/tree/main/apps/android_rpc>`_ to
#   install the TVM RPC APK on the android device. Make sure you can pass the android rpc test.
#   Then you have already registered your device. During tuning, you have to go to developer option
#   and enable "Keep screen awake during changing" and charge your phone to make it stable.
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 2 Huawei mate10 pro, 11 Raspberry Pi 3B and 2 rk3399,
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    mate10pro    2      2     0
#    rk3399       2      2     0
#    rpi3b        11     11    0
#    ----------------------------------
#
# You can register multiple devices to the tracker to accelerate the measurement in tuning.

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations. Here I use an RK3399 board
# as example. In your setting, you should modify the target and device_key accordingly.
# set :code:`use_android` to True if you use android phone.

#### DEVICE CONFIG ####

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# target = tvm.target.Target("llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib")
target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"


# Also replace this with the device key in your tracker
device_key = "wasm"

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####
network = "resnet-18"
log_file = "%s.%s.log" % (device_key, network)

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
base_path = os.path.join(curr_path) + "/turning_out"
tuning_path = os.path.join(base_path, network + ".wasm")
tuning_deploy_path = base_path + "/../../../web/dist/tuning.wasm"
# tuning_deploy_path = "/workspace/tvm/web/dist/tuning.wasm"
print("config tuning path = " + tuning_path)
print("config tuning_dst_path = " + tuning_deploy_path)

# tuning_path = "%s.%s.wasm" % (device_key, network)
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            n_parallel = 1,
            build_func = wasm_module_loader.fcompile,
        ),
        runner=autotvm.RPCRunner(
            "wasm",
            host="0.0.0.0",
            port=9190, # 端口，这里要和前面配置声明的一样
            number=1,
            repeat=10,
            module_loader = wasm_module_loader.wasm_module_loader(tuning_deploy_path),
            enable_cpu_cache_flush = True,
        ),
    ),
    "n_trial": 1,
    "use_transfer_learning": True
}

# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 1500,
#     "early_stopping": 800,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
#         runner=autotvm.RPCRunner(
#             device_key,
#             host="127.0.0.1",
#             port=9190,
#             number=5,
#             timeout=10,
#         ),
#     ),
# }

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning run longer.
#   If your device runs very slow or your conv2d operators have many GFLOPs, considering to
#   set timeout larger.
#
#   If your model has depthwise convolution, you could consider setting
#   :code:`try_spatial_pack_depthwise` be :code:`True`, which perform better than default
#   optimization in general. For example, on ARM CPU A53 2.0GHz, we find it could boost 1.6x
#   performance of depthwise convolution on Mobilenet V1 model.

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        print("===============tune_tasks prefix ===========> "+ prefix)
        # create tuner
        print("======> current tuner is "+ tuner)
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        print("||||||||||||||||||||||||||||||||||||    before tuner_obj.tune .... ||||||||||||||||||||||||||||||||||||")

        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
        print("|||||||||||||||||||||||||||||||||||| after tuner_obj.tune .... ||||||||||||||||||||||||||||||||||||")


    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)
    print("tune tasks end...")

########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def prepare_test_libs(mod,params,network,base_path):
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

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, _ = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    if not os.access(tuning_deploy_path, os.R_OK):
        print("origin wasm not set ... generating... ")
        prepare_test_libs(mod,params,network,base_path)
        
        from shutil import copyfile
        # os.unlink(tuning_deploy_path)

        wasm_path = os.path.join(base_path, f"{network}.wasm")

        copyfile(wasm_path, tuning_deploy_path)
    

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)

        lib.export_library(tuning_path, emcc.create_tvmjs_wasm)
        with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
            f_graph_json.write(graph)
        with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
            f_params.write(relay.save_param_dict(params))
        print("output: " + tuning_path)

tune_and_evaluate(tuning_option)
