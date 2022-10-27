# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
if torch.__version__>="1.8.0":
    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
    except:
        print('WARNING! torch_npu is not imported.. Please using without npu..')
from commons import print_separator
from commons import initialize_distributed
from megatron import mpu
import sys
import os

def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(torch._C, '_npu_setRNGState') and callable(torch._C._npu_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                torch._C._npu_setRNGState(new_state)
    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('npu')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('npu', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.npu.current_device()
            default_generator = torch.npu.default_generators[idx]
            default_generator.set_state(new_state)

    torch.npu._lazy_call(cb)

for k in sys.modules:
    if k.startswith('megatron'):
        for target in ['_set_cuda_rng_state']:
            if getattr(sys.modules[k], '_set_cuda_rng_state', None):
                setattr(sys.modules[k], '_set_cuda_rng_state', _set_cuda_rng_state)


def test_set_cuda_rng_state(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing set_rng_state with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    size = 123
    seed = 1234
    torch.npu.manual_seed(1234)
    tensor = torch.npu.FloatTensor(size)

    # Get the state
    rng_state = torch.npu.get_rng_state()
    rng_state_copy = rng_state.clone()

    # Do some stuff.
    for _ in range(5):
        result_1 = torch.bernoulli(tensor, p=0.5)

    assert rng_state.sub(rng_state_copy).max() == 0
    assert torch.npu.get_rng_state().sub(rng_state_copy).max() > 0

    # State should be different.
    new_rng_state = torch.npu.get_rng_state()
    max_diff = new_rng_state.sub(rng_state).max()
    print('   max diff in rng state (should be non-zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), max_diff))
    assert max_diff > 0

    # Reset the rng state and do the same stuff.
    _set_cuda_rng_state(rng_state)
    for _ in range(5):
        torch.bernoulli(tensor, p=0.5)
    _set_cuda_rng_state(rng_state)
    for _ in range(5):
        result_2 = torch.bernoulli(tensor, p=0.5)

    # Results should be the same
    error = result_2.sub(result_1).abs().max()
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Input state should have remained intact.
    error = rng_state.sub(rng_state_copy).max()
    print('   max error in rng state (should be zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), error))
    assert error == 0

    # Reset groups
    mpu.destroy_model_parallel()

    #torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_cuda_rng_tracker(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cuda rng tracker with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed_1 = 1234
    seed_2 = 4321
    size = [12, 21]
    tensor = torch.npu.FloatTensor(*size)

    # Set to seed_1 and generate two tensors.
    torch.npu.manual_seed(seed_1)
    target_11 = torch.bernoulli(tensor, p=0.5)
    target_12 = torch.bernoulli(tensor, p=0.5)

    # Set to seed_2 and generate two tensors.
    torch.npu.manual_seed(seed_2)
    target_21 = torch.bernoulli(tensor, p=0.5)
    target_22 = torch.bernoulli(tensor, p=0.5)

    # Now if we interleave seed_1 and seed_2,
    # we should still get the same tensors
    torch.npu.manual_seed(seed_1)
    mpu.get_cuda_rng_tracker().add('test', seed_2)

    # torch.randn(size, out=tensor)
    result_11 = torch.bernoulli(tensor, p=0.5)

    with mpu.get_cuda_rng_tracker().fork('test'):
        result_21 = torch.bernoulli(tensor, p=0.5)

    result_12 = torch.bernoulli(tensor, p=0.5)

    with mpu.get_cuda_rng_tracker().fork('test'):
        result_22 = torch.bernoulli(tensor, p=0.5)

    diff = result_11.sub(result_21).abs().max()
    diff = min(diff, result_12.sub(result_22).abs().max())
    print('   max diff in generated tensors (should be non-zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), diff))
    assert diff > 1.0e-6
    error = max(result_11.sub(target_11).abs().max(),
                result_12.sub(target_12).abs().max())
    error = max(error, result_21.sub(target_21).abs().max())
    error = max(error, result_22.sub(target_22).abs().max())
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset the tracker
    mpu.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    #torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_model_parallel_cuda_manual_seed(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing model parallel cuda manual seed with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    mpu.model_parallel_cuda_manual_seed(12345)
    assert torch.npu.initial_seed() == 12345
    with mpu.get_cuda_rng_tracker().fork():
        assert torch.npu.initial_seed() == (12345 + 2718 +
                                             mpu.get_tensor_model_parallel_rank())

    # Reset the tracker
    mpu.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    #torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test set rng state')
        test_set_cuda_rng_state(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test cuda rng tracker')
        test_cuda_rng_tracker(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test model parallel cuda manual seed')
        test_model_parallel_cuda_manual_seed(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
