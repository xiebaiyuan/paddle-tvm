import os
import numpy as np
import tvm
from tvm.contrib import emcc
import contextlib
from tvm import nd, rpc as _rpc

def request_remote(device_key, host=None, port=None, priority=1, timeout=10):
    """Request a remote session
    Parameters
    ----------
    device_key: string
        The device key of registered device in tracker
    host: host, optional
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST"
    port: int, optional
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT"
    priority: int, optional
        The priority of this request, larger is more prior
    timeout: float, optional
        The timeout of this session (units: second)
    Returns
    ------
    session: RPCSession
    """
    # connect to the tracker
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])
    tracker = _rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout, max_retry=1, session_constructor_args=["rpc.WasmSession", 1])
    return tracker, remote

def wasm_module_loader(tuning_deploy_path, pre_load_function=None):
    print("wasm_module_loader in ")
    print("tuning_deploy_path: " + tuning_deploy_path)
    """Returns a default function that can be passed as module_loader to run_through_rpc.
    Parameters
    ----------
    pre_load_function : Optional[Function[tvm.rpc.Session, tvm.runtime.Module]]
        Invoked after a session is established and before the default code-loading RPC calls are
        issued. Allows performing pre-upload actions, e.g. resetting the remote runtime environment.
    Returns
    -------
    ModuleLoader :
        A function that can be passed as module_loader to run_through_rpc.
    """
    @contextlib.contextmanager
    def default_module_loader_mgr(remote_kwargs, build_result):
        print("=============== default_module_loader_mgr ============")
        # print("build_result : " + build_result)

        tracker, remote = request_remote(**remote_kwargs)
        if pre_load_function is not None:
            pre_load_function(remote, build_result)
        from shutil import copyfile
        os.unlink(tuning_deploy_path)
        copyfile(build_result.filename, tuning_deploy_path)
        # print("=============== UPLOADING ============")
        remote.upload(tuning_deploy_path)
        # close connection 这里需要替换 文件之后，重新生效
        tracker.close()
        # print("=============== CLOSE ============")
        # reconnection again
        tracker, remote = request_remote(**remote_kwargs)
        # print("=============== CONNECTED ============", remote_kwargs)
        if pre_load_function is not None:
            pre_load_function(remote, build_result)
        try:
            systemModuleLib = remote.system_lib()
            systemModuleLib.entry_name = "default_function"
            yield remote, systemModuleLib
        except any:
            print("Unexpected error:") 
        finally:
            # clean up remote files
            remote.remove(build_result.filename)
            remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
            remote.remove("")
            os.unlink(build_result.filename)
    return default_module_loader_mgr
# custom localbuild function
def fcompile(*args):
    emcc.create_tvmjs_wasm(args[0], args[1])
fcompile.output_format = "wasm"
fcompile.object_format = "bc"