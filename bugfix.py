import copy
import math
import os
import sys
import time
import numpy as np
import numbers
import torch
import torch_npu
import apex
import megatron

from functools import wraps
from torch import _C
from torch_npu.npu import _lazy_call, device as device_ctx_manager
from megatron import print_rank_0, get_args
from megatron.core import mpu
from megatron.data.gpt_dataset import _num_tokens, _num_epochs, _build_shuffle_idx, _build_doc_idx
from megatron.model import ModelType
from megatron.model.module import conversion_helper
from megatron.model.fused_layer_norm import MixedFusedLayerNorm, HAVE_PERSIST_LAYER_NORM
from megatron.core.utils import make_viewless_tensor
from megatron.schedules import custom_backward, dummy_handler, forward_step, get_num_microbatches
from megatron.initialize import _warmup_jit_function


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


# deprecated
def wrapper_dist(fn):
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


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (torch.nn.parameter.Parameter, torch.autograd.Variable)):
            val_typecheck = val.data
        if val_typecheck.dtype in [torch.float16, torch.bfloat16]:
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


# deprecated
def _set_cuda_rng_state(new_state, device=-1):
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.npu.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def _build_index_mappings(name, data_prefix, documents, sizes,
                          num_samples, seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}sl'.format(seq_length)
    _filename += '_{}s'.format(seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if int(os.environ['LOCAL_RANK']) == 0:
        if (not os.path.isfile(doc_idx_filename)) or \
                (not os.path.isfile(sample_idx_filename)) or \
                (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # For the last epoch, decide whether include the entire epoch
            # in the global shuffle or not.

            # If we need only one epoch, then separating last epoch  does
            # not mean anything.
            if num_epochs == 1:
                separate_last_epoch = False
                print(' > only one epoch required, setting '
                      'separate_last_epoch to False', flush=True)

            else:
                # Get the number of samples for the last epoch
                num_samples_from_epochs_minus_one = (
                                                            (num_epochs - 1) * tokens_per_epoch - 1) // seq_length
                last_epoch_num_samples = num_samples - \
                                         num_samples_from_epochs_minus_one
                assert last_epoch_num_samples >= 0, \
                    'last epoch number of samples should be non-negative.'
                num_samples_per_epoch = (tokens_per_epoch - 1) // seq_length
                assert last_epoch_num_samples < (num_samples_per_epoch + 1), \
                    'last epoch number of samples exceeded max value.'
                # If we have less than 80% of the samples for the last epoch,
                # seperate out the epoch and treat it differently.
                # Note: the 80% number is just based on common sense and can
                # be adjusted if needed.
                separate_last_epoch = (last_epoch_num_samples <
                                       int(0.80 * num_samples_per_epoch))
                if separate_last_epoch:
                    string = ' > last epoch number of samples ({}) is smaller ' \
                             'than 80% of number of samples per epoch ({}), ' \
                             'setting separate_last_epoch to True'
                else:
                    string = ' > last epoch number of samples ({}) is larger ' \
                             'than 80% of number of samples per epoch ({}), ' \
                             'setting separate_last_epoch to False'
                print(string.format(last_epoch_num_samples,
                                    num_samples_per_epoch), flush=True)

            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng,
                                     separate_last_epoch)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save doc-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            # First compile and then import.
            from megatron.data import helpers
            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32
            sample_idx = helpers.build_sample_idx(sizes, doc_idx, seq_length,
                                                  num_epochs, tokens_per_epoch)
            # sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
            #                               num_epochs, tokens_per_epoch)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save sample-idx mapping '
                         '(seconds): {:4f}'.format(time.time() - start_time))
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            if separate_last_epoch:
                num_samples_ = num_samples_from_epochs_minus_one
            else:
                num_samples_ = sample_idx.shape[0] - 1
            shuffle_idx = _build_shuffle_idx(num_samples_,
                                             sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
            torch.distributed.get_world_size() //
            torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading sample-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        sample_idx.shape[0]))
    print_rank_0('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def MixedFusedLayerNormInit(self, normalized_shape, eps=1e-5, no_persist_layer_norm=True, sequence_parallel=False):
    super(MixedFusedLayerNorm, self).__init__()
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
    self.normalized_shape = torch.Size(normalized_shape)
    self.eps = eps
    self.weight = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
    self.bias = torch.nn.parameter.Parameter(torch.Tensor(*normalized_shape))
    self.reset_parameters()
    self.no_persist_layer_norm = True
    self.sequence_parallel = sequence_parallel

    # set sequence parallelism flag on weight and bias parameters
    setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
    setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


def MixedFusedLayerNormForward(self, input):
    if self.no_persist_layer_norm:
        return torch.nn.functional.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        output = FastLayerNormFN.apply(input, self.weight, self.bias, self.eps)
        output = make_viewless_tensor(inp=output, requires_grad=input.requires_grad, keep_graph=True)
    return output


def FusedScaleMaskSoftmaxForward(self, input, mask):
    # [b, np, sq, sk]
    assert input.dim() == 4

    if self.input_in_float16 and self.softmax_in_fp32:
        input = input.float()
    if self.scale is not None:
        input = input * self.scale
    mask_output = self.mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    if self.input_in_float16 and self.softmax_in_fp32:
        if self.input_in_fp16:
            probs = probs.half()
        else:
            probs = probs.bfloat16()

    # probs = torch_npu.npu_scaled_masked_softmax(input, mask, self.scale, False)

    return probs


def clip_grad_norm_fp32(parameters, grads_for_norm, max_norm, norm_type=2, model_parallel_group=None):
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
    if norm_type == math.inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()
    else:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for p in parameters:
            p.grad.detach().mul_(clip_coeff)
    return total_norm


def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def clear_npu_overflow_flag():
    # TODO: async operation and 1980B compatible adaptation
    float_status = torch.zeros(8).npu()
    result = torch.npu_clear_float_status(float_status)


def get_npu_overflow_flag():
    # TODO: async operation and 1980B compatible adaptation
    float_status = torch.zeros(8).npu()
    result = torch.npu_get_float_status(float_status)
    if float_status.cpu()[0] != 0:
        return True
    else:
        return False


def set_npu_overflow_flag():
    # TODO: async operation and 1980B compatible adaptation
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

    overflow_flag = get_npu_overflow_flag()
    overflow_flag_all = overflow_flag or overflow_flag_all
    if overflow_flag_all:
        set_npu_overflow_flag()
    return forward_data_store


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        # torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
    _warmup_jit_function()


def _unscale_main_grads_and_check_for_nan(self):
    main_grads = self._collect_main_grad_data_for_unscaling()
    self.found_inf.fill_(0.0)
    torch._amp_foreach_non_finite_check_and_unscale_(main_grads, self.found_inf, self.grad_scaler.inv_scale)
    torch.distributed.all_reduce(self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group())
    torch.distributed.all_reduce(self.found_inf, op=torch.distributed.ReduceOp.MAX, group=mpu.get_data_parallel_group())
    found_inf_flag = (self.found_inf.item() > 0)
    return found_inf_flag


os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
torch.Tensor.type = wrapper_type(torch.Tensor.type)
torch.distributed.all_reduce = wrapper_dist(torch.distributed.all_reduce)

megatron.optimizer.Adam = torch.optim.AdamW
megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state
megatron.data.gpt_dataset._build_index_mappings = _build_index_mappings
megatron.model.module.fp32_to_float16 = fp32_to_float16
megatron.model.module.float16_to_fp32 = float16_to_fp32

megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__ = MixedFusedLayerNormInit
megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward = MixedFusedLayerNormForward
megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward = FusedScaleMaskSoftmaxForward

megatron.optimizer.clip_grads.clip_grad_norm_fp32 = clip_grad_norm_fp32
megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan = _unscale_main_grads_and_check_for_nan


megatron.initialize._compile_dependencies = _compile_dependencies

megatron.schedules.backward_step = backward_step
megatron.schedules.forward_backward_no_pipelining = forward_backward_no_pipelining

for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'set_jit_fusion_options'):
        setattr(v, 'set_jit_fusion_options', set_jit_fusion_options)

    if 'megatron' in k and hasattr(v, 'clip_grad_norm_fp32'):
        setattr(v, 'clip_grad_norm_fp32', clip_grad_norm_fp32)
