from functools import reduce

import torch
import megatron.optimizer
from megatron.core import tensor_parallel


def DistributedOptimizerInit(self, optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                             use_contiguous_buffers_in_local_ddp, fp16, bf16, params_dtype, grad_scaler, models):
    super(megatron.optimizer.distrib_optimizer.DistributedOptimizer, self).__init__(
        optimizer, clip_grad, log_num_zeros_in_grad,
        params_have_main_grad, use_contiguous_buffers_in_local_ddp,
        fp16, bf16, params_dtype, grad_scaler, models)

    # Verify that contiguous buffers are being used.
    # - Note: this should already be checked in arguments.py.
    assert use_contiguous_buffers_in_local_ddp

    # Model grad buffer ranges.
    self.model_gbuf_ranges = []
    for model_index, model in enumerate(self.models):
        self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model))
    self.model_param_gbuf_map = \
        self.build_model_param_gbuf_map(self.model_gbuf_ranges)

    # Optimizer ranges.
    self.opt_group_ranges = self.build_optimizer_group_ranges(
        self.optimizer.param_groups,
        self.model_gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self.build_model_and_main_param_groups(self.model_gbuf_ranges,
                                               self.model_param_gbuf_map,
                                               self.opt_group_ranges)

    # Initialize param buffers.
    # - These are views on the DDP model's grad buffers, that share
    #   storage & have their own dtype. This is safe because the param
    #   dtype size is always <= grad dtype size.
    self.param_buffers = []
    for model_index, model in enumerate(self.models):
        current_param_buffers = {}
        for dtype, grad_buffer in model._grad_buffers.items():
            # create NPU tensor with set_() instead of tensor.storage()._untyped()
            param_buffer = torch.tensor(torch.flatten(grad_buffer.data),  # grad_buffer.data.storage()._untyped(),
                                        dtype=params_dtype,
                                        device=grad_buffer.data.device)

            param_buffer = param_buffer[:grad_buffer.numel_padded]
            current_param_buffers[dtype] = param_buffer
        self.param_buffers.append(current_param_buffers)

    # Update optimizer groups.
    # - Also, leverage state_dict() and load_state_dict() to
    #   recast preexisting per-param state tensors.
    self.optimizer.param_groups = \
        [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())


def build_model_and_main_param_groups(cls,
                                      model_gbuf_ranges,
                                      param_gbuf_map,
                                      opt_group_ranges):
    """
    Create main parameter groups needed for the optimizer step.

    These groups encompass both: 1) groups used by this class, for
    reducing/gather, and 2) groups used by the inner optimizer for the
    parameter update. Given that the conceptual grad buffer partitioning
    (created in earlier method) doesn't respect parameter boundaries,
    the optimizer operates on shards of the model parameters, rather than
    the full parameters.
    """

    # Parameter groups:
    #   model_float16_groups: original float16 parameters
    #   model_fp32_groups: original fp32 parameters
    #   shard_float16_groups: shards of original float16 parameters
    #   shard_fp32_groups: shards of original fp32 parameters
    #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
    model_float16_groups = []
    model_fp32_groups = []
    shard_float16_groups = []
    shard_fp32_groups = []
    shard_fp32_from_float16_groups = []

    # Allocate (or slice) each group's param shard.
    for group_index, group_range in enumerate(opt_group_ranges):

        # Params of this group.
        model_float16_params_this_group = []
        model_fp32_params_this_group = []
        shard_float16_params_this_group = []
        shard_fp32_params_this_group = []
        shard_fp32_from_float16_params_this_group = []
        model_float16_groups.append(model_float16_params_this_group)
        model_fp32_groups.append(model_fp32_params_this_group)
        shard_float16_groups.append(shard_float16_params_this_group)
        shard_fp32_groups.append(shard_fp32_params_this_group)
        shard_fp32_from_float16_groups.append(
            shard_fp32_from_float16_params_this_group)

        for model_param in group_range["params"]:

            assert model_param.requires_grad

            model_index, dtype = param_gbuf_map[model_param]
            gbuf_range = model_gbuf_ranges[model_index][dtype]
            param_range = gbuf_range["param_map"][model_param]["param"]

            # fp16, bf16 params.
            if model_param.type() in ['torch.cuda.HalfTensor',
                                      'torch.cuda.BFloat16Tensor',
                                      'torch.npu.BFloat16Tensor']:

                # Clone model -> main.
                shard_model_param = model_param.detach().view(-1) \
                    [param_range.start:param_range.end]
                shard_main_param = shard_model_param.clone().float()
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_main_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared
                    shard_main_param.shared = model_param.shared

                # Add to group.
                model_float16_params_this_group.append(model_param)
                shard_float16_params_this_group.append(shard_model_param)
                shard_fp32_from_float16_params_this_group.append(shard_main_param)

            # fp32 params.
            elif model_param.type() == 'torch.cuda.FloatTensor':
                shard_model_param = model_param.view(-1) \
                    [param_range.start:param_range.end]
                model_fp32_params_this_group.append(model_param)
                shard_fp32_params_this_group.append(shard_model_param)
                tensor_parallel.copy_tensor_model_parallel_attributes(
                    shard_model_param, model_param)
                if hasattr(model_param, 'shared'):
                    shard_model_param.shared = model_param.shared

            else:
                raise TypeError('Wrapped parameters must be one of '
                                'torch.cuda.FloatTensor,  '
                                'torch.cuda.HalfTensor, or '
                                'torch.cuda.BFloat16Tensor. '
                                'torch.npu.BFloat16Tensor. '
                                'Received {}'.format(param.type()))

        # Update optimizer's params.
        group_range["orig_group"]["params"] = [
            *shard_fp32_params_this_group,
            *shard_fp32_from_float16_params_this_group,
        ]

    return (
        model_float16_groups,
        model_fp32_groups,
        shard_float16_groups,
        shard_fp32_groups,
        shard_fp32_from_float16_groups,
    )


megatron.optimizer.distrib_optimizer.DistributedOptimizer.__init__ = DistributedOptimizerInit
megatron.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups = build_model_and_main_param_groups
