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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""PaddlePaddle: PaddlePaddle Exchange frontend for Relay."""
from __future__ import absolute_import

import copy
import warnings
import numpy as np
import tvm
import argparse
import ast
import sys
import os
import paddle.fluid as fluid
import os
import six
import paddle
import numpy as np
from paddle.fluid.framework import Variable
from paddle2onnx.utils import check_model, logging
from paddle2onnx.graph import PaddleGraph, ONNXGraph
from paddle2onnx.passes import PassManager

from tvm.ir import IRModule
from tvm.relay.frontend.onnx import GraphProto
from tvm.topi.utils import get_const_tuple
from six import text_type as _text_type

from paddle2onnx.utils import logging


from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import vision as _vision
from .. import loops as _loops
from .. import ty as _ty

from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels, infer_value, fold_constant
from .common import infer_type, get_name

__all__ = ["from_paddlepaddle"]


def from_paddlepaddle(model_dir,shape = None, save_file=None, dtype="float32", model_filename=None, params_filename=None,opset_version=9,enable_onnx_checker=False):
    print("from_paddlepaddle ==== >>>>>")

    # if len(sys.argv) < 2:
    # logging.info("Use \"paddle2onnx -h\" to print the help information")
    # logging.info(
    #         "For more information, please follow our github repo below:")
    # logging.info("Github: https://github.com/PaddlePaddle/paddle2onnx.git")
    # return

    # parser = arg_parser()
    # args = parser.parse_args()

    # if args.version:
    #     import paddle2onnx
    #     logging.info("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0".
    #                  format(paddle2onnx.__version__))
    #     return

    assert model_dir is not None, "model_dir should be defined while translating paddle model to onnx"
    
    
    # assert save_file is not None, "save_file should be defined while translating paddle model to onnx"
    
    
    model = loadfileprogram2onnx(
        model_dir,
        save_file,
        model_filename,
        params_filename,
        opset_version=opset_version,
        enable_onnx_checker=enable_onnx_checker)


    

    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except onnx.onnx_cpp2py_export.checker.ValidationError as e:  # pylint: disable=c-extension-no-member
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    freeze_params = True
    
    g = GraphProto(shape, dtype, freeze_params)
    graph = model.graph
    opset = None
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
        mod, params = g.from_onnx(graph, opset)
    return mod, params



def from_paddlepaddle_onnx(model, shape=None, dtype="float32", opset=None, freeze_params=False):
    print("from_paddlepaddle ==== >>>>>")
    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except onnx.onnx_cpp2py_export.checker.ValidationError as e:  # pylint: disable=c-extension-no-member
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    g = GraphProto(shape, dtype, freeze_params)
    graph = model.graph
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
        mod, params = g.from_onnx(graph, opset)
    return mod, params

def export_onnx(paddle_graph,
                save_file=None,
                opset_version=9,
                enable_onnx_checker=False,
                verbose=False):
    onnx_graph = ONNXGraph.build(paddle_graph, opset_version, verbose)
    onnx_graph = PassManager.run_pass(onnx_graph, ['inplace_node_pass'])

    onnx_proto = onnx_graph.export_proto(enable_onnx_checker)
    

    if save_file == None:
        return onnx_proto
    path, _ = os.path.split(save_file)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_file, 'wb') as f:
        f.write(onnx_proto.SerializeToString())
    logging.info("ONNX model saved in {}".format(save_file))

    return onnx_proto


def program2onnx(program,
                 scope,
                 save_file=None,
                 feed_var_names=None,
                 target_vars=None,
                 opset_version=9,
                 enable_onnx_checker=False,
                 **configs):
    from paddle import fluid
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()
    if isinstance(program, paddle.fluid.framework.Program):
        if feed_var_names is not None:
            if isinstance(feed_var_names, six.string_types):
                feed_var_names = [feed_var_names]
            else:
                if not (bool(feed_var_names) and all(
                        isinstance(name, six.string_types)
                        for name in feed_var_names)):
                    raise TypeError("'feed_var_names' should be a list of str.")

        if target_vars is not None:
            if isinstance(target_vars, Variable):
                target_vars = [target_vars]
            else:
                if not (bool(target_vars) and
                        all(isinstance(var, Variable) for var in target_vars)):
                    raise TypeError(
                        "'target_vars' should be a list of variable.")

        paddle_graph = PaddleGraph.build_from_program(program, feed_var_names,
                                                      target_vars, scope)
        return export_onnx(paddle_graph, save_file, opset_version, enable_onnx_checker)
    else:
        raise TypeError(
            "the input 'program' should be 'Program', but received type is %s."
            % type(program))

def loadfileprogram2onnx(model_dir,
                 save_file=None,
                 model_filename=None,
                 params_filename=None,
                 opset_version=9,
                 enable_onnx_checker=False):
    try:
        import paddle
    except:
        logging.error(
            "paddlepaddle not installed, use \"pip install paddlepaddle\"")

    v0, v1, v2 = paddle.__version__.split('.')
    if v0 == '0' and v1 == '0' and v2 == '0':
        logging.warning("You are use develop version of paddlepaddle")
    elif int(v0) <= 1 and int(v1) < 8:
        raise ImportError("paddlepaddle>=1.8.0 is required")

    import paddle2onnx as p2o
    # convert model save with 'paddle.fluid.io.save_inference_model'
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()
    exe = fluid.Executor(fluid.CPUPlace())
    if model_filename is None and params_filename is None:
        [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(
            model_dir, exe)
    else:
        [program, feed_var_names, fetch_vars] = fluid.io.load_inference_model(
            model_dir,
            exe,
            model_filename=model_filename,
            params_filename=params_filename)
    return program2onnx(
        program,
        fluid.global_scope(),
        save_file,
        feed_var_names=feed_var_names,
        target_vars=fetch_vars,
        opset_version=opset_version,
        enable_onnx_checker=enable_onnx_checker)


# def main():
#     if len(sys.argv) < 2:
#         logging.info("Use \"paddle2onnx -h\" to print the help information")
#         logging.info(
#             "For more information, please follow our github repo below:")
#         logging.info("Github: https://github.com/PaddlePaddle/paddle2onnx.git")
#         return

#     parser = arg_parser()
#     args = parser.parse_args()

#     if args.version:
#         import paddle2onnx
#         logging.info("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0".
#                      format(paddle2onnx.__version__))
#         return

#     assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to onnx"
#     assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"
#     program2onnx(
#         args.model_dir,
#         args.save_file,
#         args.model_filename,
#         args.params_filename,
#         opset_version=args.opset_version,
#         enable_onnx_checker=args.enable_onnx_checker)
