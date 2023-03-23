import torch
import megatron.optimizer


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


megatron.optimizer.distrib_optimizer.DistributedOptimizer.__init__ = DistributedOptimizerInit
