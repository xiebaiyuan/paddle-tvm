import paddle
from tvm import relay
import tvm

paddle.enable_static()

[net_program, 
feed_target_names, 
fetch_targets] = paddle.static.load_inference_model('MobileNetV1/MobileNetV1',paddle.static.Executor(paddle.fluid.CPUPlace()))

shape_dict = {'image': [1,3,224,224]}
mod, params = relay.frontend.from_paddle(net_program, shape_dict=shape_dict)

with tvm.transform.PassContext(opt_level=1):
    interp = relay.build_module.create_executor("graph",mod,tvm.cpu(0), 'llvm')

