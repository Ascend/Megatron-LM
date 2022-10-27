import sys
import time
import math
import copy
import numpy as np
import torch
import torch_npu
from functools import wraps

# ======================
# torch
# ======================

# INPLACE.1: torch.cuda.get_rng_state
torch.cuda.get_rng_state = torch.get_rng_state
torch.cuda.set_rng_state = torch.set_rng_state


# INPLACE.2: torch.Tensor.type()
def wrapper_type(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        output = fn(*args, **kwargs)
        if isinstance(output, str):
            if output == 'torch.npu.FloatTensor':
                output = 'torch.cuda.FloatTensor'
            elif output == 'torch.npu.HalfTensor':
                output = 'torch.cuda.HalfTensor'
        return output

    return decorated


torch.Tensor.type = wrapper_type(torch.Tensor.type)

# INPLACE.3: torch.ditributed.xx input long --> int
from torch import distributed as dist


def wrapper_dist_long2int(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if args[0].dtype == torch.long and not kwargs.get('async_op', False):
            new_args = list(copy.deepcopy(args))
            new_args[0] = new_args[0].int()
            fn(*new_args, **kwargs)
            args[0].copy_(new_args[0].long())
            return
        return fn(*args, **kwargs)

    return wrapper


dist.all_reduce = wrapper_dist_long2int(dist.all_reduce)
dist.broadcast = wrapper_dist_long2int(dist.broadcast)
dist.send = wrapper_dist_long2int(dist.send)
dist.recv = wrapper_dist_long2int(dist.recv)

# ======================
# apex
# ======================

# INPLACE.4: apex.optimizers
import apex


class AdamW(torch.optim.Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


apex.optimizers.FusedAdam = AdamW
apex.optimizers.FusedSGD = torch.optim.SGD

# ======================
# megatron
# ======================
import megatron


# INPLACE.5: megatron.initialize._compile_dependencies
def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


megatron.initialize._compile_dependencies = _compile_dependencies

# INPLACE.6: fp32_to_float16, float16_to_fp32
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from megatron.model.module import fp32_to_float16, float16_to_fp32, conversion_helper


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


megatron.model.module.fp32_to_float16 = fp32_to_float16
megatron.model.module.float16_to_fp32 = float16_to_fp32

# INPLACE.7: MixedFusedLayerNorm
from megatron.model.fused_layer_norm import MixedFusedLayerNorm


class MixedFusedLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, no_persist_layer_norm=True, sequence_parallel=False):
        super(MixedFusedLayerNorm, self).__init__(normalized_shape, eps, no_persist_layer_norm)

        # set sequence parallelism flag on weight and bias parameters
        self.sequence_parallel = sequence_parallel
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


for k in sys.modules:
    if k.startswith('megatron.model'):
        for target in ['LayerNorm', 'MixedFusedLayerNorm']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, MixedFusedLayerNorm)

# INPLACE.8: _unscale_main_grads_and_check_for_nan
from megatron.optimizer import Float16OptimizerWithFloat16Params


def _unscale_main_grads_and_check_for_nan(self):
    # Collect main grads.
    main_grads = self._collect_main_grad_data_for_unscaling()

    # Reset found inf.
    self.found_inf.fill_(0.0)

    # Unscale and set found inf/nan
    torch._amp_foreach_non_finite_check_and_unscale_(
        main_grads, self.found_inf, self.grad_scaler.inv_scale)

    # Update across all model parallel instances.
    torch.distributed.all_reduce(self.found_inf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=self.get_model_parallel_group())

    # add data_parallel synchronize
    torch.distributed.all_reduce(self.found_inf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=self.get_data_parallel_group())

    # Check for nan.
    found_inf_flag = (self.found_inf.item() > 0)

    return found_inf_flag


Float16OptimizerWithFloat16Params._unscale_main_grads_and_check_for_nan = _unscale_main_grads_and_check_for_nan

# INPLACE.9: FusedScaleMaskSoftmax
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.enums import AttnMaskType


class FusedScaleMaskSoftmax(torch.nn.Module):
    def __init__(
            self,
            input_in_fp16,
            input_in_bf16,
            attn_mask_type,
            scaled_masked_softmax_fusion,
            mask_func,
            softmax_in_fp32,
            scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
                self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.mask_tri = None

        assert (
                self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        if torch.npu.is_available():
            return self.forward_fused_softmax(input, mask)

        return self.forward_torch_softmax(input, mask)

    def forward_fused_softmax(self, input, mask):
        if self.softmax_in_fp32:
            input = input.float()

        if self.scale is None:
            self.scale = 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            if self.mask_tri is None:
                self.mask_tri = torch.triu(torch.ones(input.shape, device=input.device), diagonal=1).bool()
            probs = torch.npu_scaled_masked_softmax(input, self.mask_tri, self.scale, False)
        else:
            probs = torch.npu_scaled_masked_softmax(input, mask, self.scale, False)

        probs = probs.half()

        return probs

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale

        if self.attn_mask_type == AttnMaskType.causal:
            mask_tri = torch.triu(torch.ones(input.shape, device=input.device), diagonal=1).bool()
            mask_output = self.mask_func(input, mask_tri)
        else:
            mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs


for k in sys.modules:
    if k.startswith('megatron.model'):
        for target in ['FusedScaleMaskSoftmax']:
            if getattr(sys.modules[k], target, None):
                setattr(sys.modules[k], target, FusedScaleMaskSoftmax)

# INPLACE.10: clip_grad_norm_fp32
from torch._six import inf
from megatron import mpu
from megatron.model.module import param_is_not_shared
from megatron.mpu.layers import param_is_not_tensor_parallel_duplicate
from megatron.optimizer.clip_grads import clip_grad_norm_fp32


def clip_grad_norm_fp32(parameters, grads_for_norm,
                        max_norm, norm_type=2,
                        model_parallel_group=None):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (group): given the nature of the distributed
            optimizer, this is passed as an argument.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()

    else:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coeff)

    return total_norm


megatron.optimizer.clip_grads.clip_grad_norm_fp32 = clip_grad_norm_fp32
megatron.optimizer.optimizer.clip_grad_norm_fp32 = clip_grad_norm_fp32

# INPLACE.11: _CUDA_RNG_STATE_TRACKER
import contextlib
from megatron.mpu.random import CudaRNGStatesTracker, _CUDA_RNG_STATE_TRACKER, _MODEL_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))
        # Get the current rng state.
        # orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        # self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        # _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        yield
        # """Fork the cuda rng state, perform operations, and exit with
        # the original state."""
        # # Check if we have added the state
        # if name not in self.states_:
        #     raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        # orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        # _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        # try:
        #     yield
        # finally:
        #     # Update the current rng state for later use.
        #     self.states_[name] = torch.cuda.get_rng_state()
        #     # And set the state to the original state we started with.
        #     _set_cuda_rng_state(orig_cuda_rng_state)


megatron.mpu.random.CudaRNGStatesTracker = CudaRNGStatesTracker
megatron.mpu.random._CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()

# INPLACE.12: _unscale_main_grads_and_check_for_nan
from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params


def _unscale_main_grads_and_check_for_nan(self):
    main_grads = []
    # fp32 params fromm float16 ones.
    for main_group in self.fp32_from_float16_groups:
        for main_param in main_group:
            if main_param.grad is not None:
                main_grads.append(main_param.grad.data)
    # Append fp32 parameters.
    for main_group in self.fp32_from_fp32_groups:
        for main_param in main_group:
            if main_param.grad is not None:
                main_grads.append(main_param.grad.data)
    # Reset found inf.
    self.found_inf.fill_(0.0)
    # Unscale and set found inf/nan
    torch._amp_foreach_non_finite_check_and_unscale_(
        main_grads, self.found_inf, self.grad_scaler.inv_scale)
    # Update across all model parallel instances.
    torch.distributed.all_reduce(self.found_inf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=mpu.get_model_parallel_group())
    torch.distributed.all_reduce(self.found_inf,
                                 op=torch.distributed.ReduceOp.MAX,
                                 group=mpu.get_data_parallel_group())

    # Check for nan.
    found_inf_flag = (self.found_inf.item() > 0)
    return found_inf_flag


Float16OptimizerWithFloat16Params._unscale_main_grads_and_check_for_nan = _unscale_main_grads_and_check_for_nan

# INPLACE.13: refine overflow flag
from megatron import schedules, get_num_microbatches, get_args, get_timers
from megatron.schedules import dummy_handler, forward_step, custom_backward
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP


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


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()

    timers = get_timers()
    timers('backward-compute').start()

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

    timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(forward_step_func, data_iterator, model,
                                   optimizer, timers, forward_only):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    losses_reduced = []
    input_tensor, output_tensor_grad = None, None
    overflow_flag_all = False
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator, model,
                                         input_tensor, losses_reduced)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            overflow_flag = get_npu_overflow_flag()
            overflow_flag_all = overflow_flag or overflow_flag_all
    output_tensor = forward_step(forward_step_func, data_iterator, model,
                                 input_tensor, losses_reduced)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    overflow_flag = get_npu_overflow_flag()
    overflow_flag_all = overflow_flag or overflow_flag_all

    if overflow_flag_all:
        set_npu_overflow_flag()
    return losses_reduced


schedules.forward_backward_no_pipelining = forward_backward_no_pipelining

# INPLACE.14: remove dropout in ParallelTransformerLayer
from megatron.model.transformer import ParallelTransformerLayer, bias_dropout_add_fused_train, \
    bias_dropout_add_fused_inference, get_bias_dropout_add
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType


def forward(self, hidden_states, attention_mask,
            encoder_output=None, enc_dec_attn_mask=None,
            inference_params=None):
    # hidden_states: [b, s, h]

    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output, attention_bias = \
        self.self_attention(
            layernorm_output,
            attention_mask,
            inference_params=inference_params)

    # Residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    # jit scripting for a nn.module (with dropout) is not
    # trigerring the fusion kernel. For now, we use two
    # different nn.functional routines to account for varying
    # dropout semantics during training and inference phases.
    if self.bias_dropout_fusion:
        if self.training:
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:
            bias_dropout_add_func = bias_dropout_add_fused_inference
    else:
        bias_dropout_add_func = get_bias_dropout_add(self.training)

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        layernorm_input = bias_dropout_add_func(
            attention_output,
            attention_bias.expand_as(residual),
            residual,
            0.)  # using 0. instead of self.hidden_dropout to avoid non-convergence

    # Layer norm post the self attention.
    layernorm_output = self.post_attention_layernorm(layernorm_input)

    if self.layer_type == LayerType.decoder:
        attention_output, attention_bias = \
            self.inter_attention(layernorm_output,
                                 enc_dec_attn_mask,
                                 encoder_output=encoder_output)
        # residual connection
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                0.)  # using 0. instead of self.hidden_dropout to avoid non-convergence

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

    # MLP.
    mlp_output, mlp_bias = self.mlp(layernorm_output)

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = layernorm_input

    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        output = bias_dropout_add_func(
            mlp_output,
            mlp_bias.expand_as(residual),
            residual,
            0.)

    return output


ParallelTransformerLayer.forward = forward
