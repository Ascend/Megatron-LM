import torch
import megatron
from megatron import get_args
from megatron.core import mpu
from megatron.model import ModelType
from megatron.schedules import custom_backward, dummy_handler, forward_step, get_num_microbatches
from . import FLAG_SUPPORT_INF_NAN


def clear_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch.npu_clear_float_status(float_status)


def get_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch.npu_get_float_status(float_status)
    if float_status.cpu()[0] != 0:
        return True
    else:
        return False


def set_npu_overflow_flag():
    torch.tensor([65504]).half().npu() + 100


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad, timers):
    args = get_args()
    if timers is not None:
        timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if not FLAG_SUPPORT_INF_NAN:
        clear_npu_overflow_flag()
    if output_tensor_grad[0] is None:
        output_tensor = optimizer.scale_loss(output_tensor[0])
    custom_backward(output_tensor[0], output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if timers is not None:
        timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(forward_step_func, data_iterator, model, optimizer, timers, forward_only,
                                   collect_non_loss_data=False):
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        context_handler = model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    overflow_flag_all = False
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, input_tensor, forward_data_store,
                                         timers, collect_non_loss_data)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad, timers)

            if not FLAG_SUPPORT_INF_NAN:
                overflow_flag = get_npu_overflow_flag()
                overflow_flag_all = overflow_flag or overflow_flag_all

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, input_tensor, forward_data_store,
                                 timers, collect_non_loss_data)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor,
                      output_tensor_grad, timers)

    if not FLAG_SUPPORT_INF_NAN:
        overflow_flag = get_npu_overflow_flag()
        overflow_flag_all = overflow_flag or overflow_flag_all
        if overflow_flag_all:
            set_npu_overflow_flag()
    return forward_data_store


def deallocate_output_tensor(out):
    if out is None:
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    with torch.no_grad():
        out.set_(torch.empty((1,), device=out.device, dtype=out.dtype))


megatron.schedules.backward_step = backward_step
megatron.schedules.forward_backward_no_pipelining = forward_backward_no_pipelining
megatron.schedules.deallocate_output_tensor = deallocate_output_tensor
