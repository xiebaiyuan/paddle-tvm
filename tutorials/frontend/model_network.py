#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can either load some pre-defined network from :code:`relay.testing`
# or building :any:`relay.testing.resnet` with relay.
# We can also load models from MXNet, ONNX and TensorFlow.
#
# In this tutorial, we choose resnet-18 as tuning example.
import tvm
import paddle
from tvm import relay
batch_size = 1
dtype = "float32"
input_name = "image"
        
def get_lwk_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""

    model_path = '/data/lmk_demo/lmk_demo/data/mv6s_shift_aug_infer'
    print(model_path)

    # now you have paddle detection on disk

    paddle.enable_static()

    [net_program, 
    feed_target_names, 
    fetch_targets] = paddle.static.load_inference_model(model_path, model_filename= 'model',params_filename='params',executor=paddle.static.Executor(paddle.fluid.CPUPlace()))
    shape_dict = {input_name: [batch_size,3,64,64]}
    mod, params = relay.frontend.from_paddle(net_program, shape_dict)
    output_shape0 = [batch_size,300]
    output_shape1 = [batch_size,2]
    return mod, params, [batch_size,3,64,64], output_shape0, output_shape1
