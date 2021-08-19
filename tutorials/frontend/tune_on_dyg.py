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
.. _tune_relay_x86:
Auto-tuning a Convolutional Network for x86 CPU
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_
This is a tutorial about how to tune convolution neural network
for x86 CPU.
Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""
import os
import numpy as np
import tvm
from tvm import relay, autotvm
# wasm
from tvm.contrib import emcc
import wasm_module_loader
import model_network
# import tune_tasks
target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
network = "/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer/tune/"
input_name = "image"

log_file = "%s.wasm.log" % network
graph_opt_sch_file = "%s_graph_opt.log" % network
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
            repeat=100,
            module_loader = wasm_module_loader.wasm_module_loader(),
            enable_cpu_cache_flush = True,
        ),
    ),
    "n_trial": 1,
    "use_transfer_learning": True
}
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape0,out_shape1 = model_network.get_lwk_network(network, batch_size=1)
  
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )
  
    tasks_filter = []
    for item in tasks:
        print(item.args)
        if "depthwise" in item.name:
            tasks_filter.append(item)
    # run tuning tasks
    print("Tuning...")
    current_tasks = tasks_filter
    # print(current_tasks)
    task_total_count = len(tasks_filter)
    # current_tasks = tasks_filter[:20] # 上次结束的位置。这里根据实际的情况，使用全部或是部分的 task
    current_tasks = tasks_filter[:(task_total_count - 6)] # 上次结束的位置
    # tune_tasks.tune_tasks(current_tasks, **tuning_opt)
    # compile kernels with history best records
    
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        base_path = os.path.join(curr_path) + "/turning_out"
        turning_path = os.path.join(base_path, network + ".wasm")
        lib.export_library(turning_path, emcc.create_tvmjs_wasm)
        with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
            f_graph_json.write(graph)
        with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
            f_params.write(relay.save_param_dict(params))
        print("output: " + turning_path)


def tune_and_evaluate_graph(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    # mod, params, input_shape, out_shape = model_network.get_network(network, batch_size=1)
    mod, params, input_shape, out_shape0,out_shape1 = model_network.get_lwk_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )
    # 这里可以根据实际情况把相关的 op 融合进来
    # tasks_filter = []
    # for item in tasks:
    #     print(item.name)
    #     # if "depthwise" not in item.name:
    #     if "depthwise" in item.name:
    #         tasks_filter.append(item)
    # run tuning tasks
    print("Tuning...")
    # tune_tasks.tune_tasks(tasks, **tuning_opt)
    print(input_shape, input_name)
    # tune_tasks.tune_graph(target, 
    #     mod["main"], 
    #     input_shape, 
    #     log_file, 
    #     graph_opt_sch_file, 
    #     tuning_option["measure_option"],
    #     input_name)
    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        print("DONE")
        curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        base_path = os.path.join(curr_path) + "/turning_graph_out"
        turning_path = os.path.join(base_path, network + ".wasm")
        lib.export_library(turning_path, emcc.create_tvmjs_wasm)
        with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
            f_graph_json.write(graph)
        with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
            f_params.write(relay.save_param_dict(params))
        print("output: " + turning_path)
# auto tvm local 优化
# tune_and_evaluate(tuning_option)
tune_and_evaluate_graph(tuning_option) # autotvm global search 