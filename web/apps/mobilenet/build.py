"""Example script to create tvm wasm modules and deploy it."""

import argparse
import os
from tvm import relay
from tvm.contrib import utils, emcc
from tvm import rpc
from tvm.contrib.download import download_testdata
import tvm
from tvm._ffi import libinfo
import shutil
import logging
import json
# import tvm.ir.transform
from tvm.driver import build

curr_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))

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
    mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
    net = mod["main"]
    net = relay.Function(
        net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    )
    mod = tvm.IRModule.from_expr(net)
    return mod, params, input_shape, output_shape, net

def get_local_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    # Set the input name of the graph
    # For ONNX models, it is typically "0".
    input_name = "data"
    dtype = "float32"

    import onnx
    import caffe2

    # model_path = "./mobilenetv2-7.onnx"
    
    shape_dict = {input_name: input_shape}

    if name == "caffe":
        from caffe2.python import core, workspace
        with open("mobilenet_v2_deploy.prototxt", "r") as f:
            init_net= f.read()
        with open("mobilenet_v2.caffemodel", "rb") as f:
            predict_net = f.read()
        
        p = workspace.Predictor(init_net, predict_net)
        # mod, params = relay.frontend.from_caffe2("mobilenet_v2_deploy.prototxt", "mobilenet_v2.caffemodel", shape_dict)
        # net = mod["main"]
    elif name == "onnx":
        # 这个模型现在数据有问题
        model_path = "./mobilenet_v2.onnx"
        onnx_model = onnx.load(model_path)
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        net = mod["main"]
    elif name == "paddle":
        # 这个模型现在数据有问题
        shape_dict = {"image": input_shape}
        mobile_netv3_path="/Users/dengyuguang/Downloads/MobileNetV2_SPLIT"
        mod, params = relay.frontend.from_paddlepaddle(mobile_netv3_path, shape_dict)
        net = mod["main"]
    else:
        # 这个模型现在数据有问题
        model_path = "./mobilenet_v2.onnx"
        mod, params = relay.frontend.from_mxnet(model_path, shape_dict)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    return mod, params, input_shape, output_shape, net

def prepare_test_mobilenetlibs(base_path):
    network = "mobilenet"
    target = "llvm -mtriple=wasm32-unknown-unknown-wasm -system-lib"
    if not tvm.runtime.enabled(target):
        raise RuntimeError("Target %s is not enbaled" % target)

    batch_size = 1

    mod, params, data_shape, out_shape, func = get_local_network("paddle", batch_size)
    # mod, params, data_shape, out_shape, func = get_network("mxnet", batch_size)

    wasm_path = os.path.join(base_path, "mobilenet.wasm")
    turning_path = os.path.join(base_path, "turning.wasm")
    
    # relay 打包runtiem，模型参数，模型结构3个文件
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    lib.export_library(wasm_path, emcc.create_tvmjs_wasm)
    print("output: " + wasm_path)

    lib.export_library(turning_path, emcc.create_tvmjs_wasm)
    print("output: " + turning_path)

    with open(os.path.join(base_path, f"{network}.json"), "w") as f_graph_json:
        f_graph_json.write(graph)
    with open(os.path.join(base_path, f"{network}.params"), "wb") as f_params:
        f_params.write(relay.save_param_dict(params))

    # debugger


def build_module(opts):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_mobilenetlibs(os.path.join(curr_path) + "/dist")

def prepare_data(opts):
    """Build auxiliary data file."""
    # imagenet synset
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                          '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                          '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                        'imagenet1000_clsid_to_human.txt'])
    synset_name = 'imagenet1000_clsid_to_human.txt'
    synset_path = download_testdata(synset_url, synset_name, module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    build_dir = opts.out_dir
    with open(os.path.join(build_dir, "imagenet1k_synset.json"), "w") as f_synset:
        f_synset.write(json.dumps(synset))
    # Test image
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join(build_dir, "cat.png")
    download_testdata(image_url, image_fn)
    # # copy runtime js files.
    shutil.copyfile(libinfo.find_lib_path("tvmjs.bundle.js")[0],
                    os.path.join(build_dir, "tvmjs.bundle.js"))
    shutil.copyfile(libinfo.find_lib_path("tvmjs_runtime.wasi.js")[0],
                    os.path.join(build_dir, "tvmjs_runtime.wasi.js"))
    shutil.copyfile(os.path.join(curr_dir, "index.html"),
                    os.path.join(build_dir, "index.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default=os.path.join(curr_dir, "dist"))
    parser.add_argument("--network", default="mobilenet1.0")
    parser.add_argument('-p', '--prepare', action='store_true')

    opts = parser.parse_args()

    build_dir = os.path.abspath(opts.out_dir)
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
        opts.out_dir = build_dir

    if opts.prepare:
        prepare_data(opts)
    else:
        prepare_data(opts)
        build_module(opts)