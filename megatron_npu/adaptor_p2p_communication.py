import torch
import operator
import megatron
from functools import reduce
from megatron import get_args, core
from megatron.core import mpu
from megatron.p2p_communication import _communicate_shapes


def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next, tensor_shape, dtype_=None):
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # Some legacy inference code doesn't set the tensor shape, do so now
    # for the normal values for gpt/bert. This could be removed if inference
    # code is changed to provide tensor_shape.
    if not args.variable_seq_lengths:
        if tensor_shape is None:
            recv_prev_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
            recv_next_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = \
            _communicate_shapes(tensor_send_next,
                                tensor_send_prev,
                                recv_prev,
                                recv_next)

    override_scatter_gather_tensors_in_pipeline = False
    if args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        recv_prev_chunk_shape = reduce(operator.mul, recv_prev_shape, 1)
        recv_next_chunk_shape = reduce(operator.mul, recv_next_shape, 1)
        if recv_prev_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0 and \
                recv_next_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
            recv_prev_chunk_shape = recv_prev_chunk_shape // \
                                    mpu.get_tensor_model_parallel_world_size()
            recv_next_chunk_shape = recv_next_chunk_shape // \
                                    mpu.get_tensor_model_parallel_world_size()
        else:
            recv_prev_chunk_shape = recv_prev_shape
            recv_next_chunk_shape = recv_next_shape
            override_scatter_gather_tensors_in_pipeline = True
    else:
        recv_prev_chunk_shape = recv_prev_shape
        recv_next_chunk_shape = recv_next_shape

    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(recv_prev_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(recv_next_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        if tensor_send_next is not None:
            tensor_send_next = core.tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = core.tensor_parallel.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    if args.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                        tensor_recv_prev=tensor_recv_prev,
                                        tensor_send_next=tensor_send_next,
                                        tensor_recv_next=tensor_recv_next,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(send_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        if recv_prev:
            tensor_recv_prev = core.tensor_parallel.gather_split_1d_tensor(
                tensor_recv_prev).view(recv_prev_shape).requires_grad_()
            tensor_recv_prev = core.utils.make_viewless_tensor(tensor_recv_prev,
                                                               requires_grad=True,
                                                               keep_graph=False)

        if recv_next:
            tensor_recv_next = core.tensor_parallel.gather_split_1d_tensor(
                tensor_recv_next).view(recv_next_shape).requires_grad_()
            tensor_recv_next = core.utils.make_viewless_tensor(tensor_recv_next,
                                                               requires_grad=True,
                                                               keep_graph=False)

    return tensor_recv_prev, tensor_recv_next


megatron.p2p_communication._communicate = _communicate
