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
        import bugfix

        option = {}
        option["ACL_OP_COMPILER_CACHE_MODE"] = "disable"  # cache功能启用
        option["ACL_OP_COMPILER_CACHE_DIR"] = "./cache"  # cache所在文件夹
        print("option:", option)
        torch.npu.set_option(option)
    except:
        print('WARNING! torch_npu is not imported.. Please using without npu..')
from megatron.mpu import layers
from commons import set_random_seed
from commons import print_separator
from commons import initialize_distributed
from megatron import mpu, get_global_memory_buffer
from torch.nn.parameter import Parameter
import torch.nn.init as init
import random
import sys
import torch.nn as nn
import math
import torch.nn.functional as F
from megatron.initialize import initialize_megatron

initialize_megatron(
    args_defaults = {
        'micro_batch_size': 1,
        'num_layers': 1,
        'hidden_size': 32,
        'num_attention_heads': 16,
        'max_position_embeddings': 128,
        'encoder_seq_length': 32,
        'use_cpu_initialization': True
    })

def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def _initialize_affine_weight(weight, output_size, input_size,
                              per_partition_size, partition_dim, init_method,
                              stride=1, return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    # If we only use 1 process for model parallelism, bypass scatter.
    world_size = mpu.get_tensor_model_parallel_world_size()
    if world_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=weight.dtype,
                                requires_grad=False)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class BertParallelTransformerOutput(torch.nn.Module):
    """The output layer used after self attention and intermediate
    parts of transformer layer."""
    def __init__(self, input_size, output_size, dropout_prob,
                 layernorm_epsilon=1.0e-12, input_is_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerOutput, self).__init__()
        # Components.
        self.dense = mpu.RowParallelLinear(input_size,
                                           output_size,
                                           input_is_parallel=input_is_parallel,
                                           init_method=init_method)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.layernorm = nn.LayerNorm(output_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, input_tensor):
        hidden_states, bias = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        layernorm_input = hidden_states + input_tensor
        hidden_states = self.layernorm(layernorm_input)
        return hidden_states


class BertParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for Bert.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        intermediate_size: size of the intermediate state after
                           self attention. In both BERT and GPT
                           this is set to be 4 times the hidden
                           size.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        intermediate_activation_fn: activation function for output
                                    of intermediate.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
    """
    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 intermediate_activation_fn,
                 layernorm_epsilon,
                 init_method=init.xavier_normal_):
        super(BertParallelTransformerLayer, self).__init__()

        # Self attention.
        self.attention = BertParallelSelfAttention(hidden_size,
                                                   num_attention_heads,
                                                   attention_dropout_prob,
                                                   output_parallel=True,
                                                   init_method=init_method)
        # Self attention output.
        self.self_output = BertParallelTransformerOutput(
            hidden_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)
        # Intermediate.
        self.intermediate = mpu.ColumnParallelLinear(hidden_size, intermediate_size,
                                                     gather_output=False,
                                                     init_method=init_method)
        self.intermediate_activation_fn = intermediate_activation_fn
        # Output.
        self.output = BertParallelTransformerOutput(
            intermediate_size, hidden_size, output_dropout_prob,
            layernorm_epsilon=layernorm_epsilon,
            input_is_parallel=True,
            init_method=init_method)

    def forward(self, hidden_states, attention_mask):
        # [b, s, hp]
        attention_output_parallel = self.attention(hidden_states,
                                                   attention_mask)
        # [b, s, h]
        attention_self_output = self.self_output(attention_output_parallel,
                                                 hidden_states)
        # [b, s, ip]
        intermediate_output_parallel, bias = self.intermediate(attention_self_output)
        intermediate_output_parallel = self.intermediate_activation_fn(
            intermediate_output_parallel)
        # [b, s, h]
        layer_output = self.output(intermediate_output_parallel,
                                   attention_self_output)

        return layer_output


class BertParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for BERT.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        output_parallel: If true, no all-gather is done on the output and
                         the output values will be per partition.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 dropout_prob, output_parallel=False,
                 init_method=init.xavier_normal_):
        super(BertParallelSelfAttention, self).__init__()
        # Input configuration.
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.output_parallel = output_parallel
        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.dropout = torch.nn.Dropout(dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):

        # Attention heads. [b, s, hp]
        mixed_x_layer, bias = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Raw attention scores. [b, np, s, s]
        norm_factor = math.sqrt(math.sqrt(self.hidden_size_per_attention_head))
        attention_scores = torch.matmul(query_layer/norm_factor,
                                        key_layer.transpose(-1, -2)/norm_factor)
        # Apply the attention mask.
        attention_scores += attention_mask

        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        # get_cuda_rng_tracker is not supported on npu now..
        # with get_cuda_rng_tracker().fork():
        attention_probs = self.dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        if self.output_parallel:
            output = context_layer
        else:
            output = mpu.gather_from_tensor_model_parallel_region(context_layer)

        return output

class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 keep_master_weight_for_test=False):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set some detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # Divide the weight matrix along the embedding dimension.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.embedding_dim_per_partition = divide(self.embedding_dim,
                                                  world_size)

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings,
                                             self.embedding_dim_per_partition))
        self.weight.model_parallel = True
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.num_embeddings, self.embedding_dim,
            self.embedding_dim_per_partition, 1, init_method,
            stride=1, return_master_weight=False)

    def forward(self, input_):
        input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
        output_parallel = F.embedding(input_parallel, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        output = mpu.gather_from_tensor_model_parallel_region(output_parallel)
        return output


def test_parallel_embedding(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing parallel embedding with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).npu()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).npu()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).npu()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_parallel = ParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).npu()
    output = embedding_parallel(input_data)
    loss_parallel = torch.mul(output, loss_weight).sum()
    loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = mpu.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).npu()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_parallel.sub(loss_original).abs()
    print('   error in loss (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-4, 'error: {}'.format(error)

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print('   error in loss (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-4, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   hidden_size // tensor_model_parallel_size,
                                   1)[mpu.get_tensor_model_parallel_rank()]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print('   error in grad (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-4, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // tensor_model_parallel_size,
                                   0)[mpu.get_tensor_model_parallel_rank()]
    error = embedding_vocab_parallel.weight.grad.sub(
        weight_grad_orig).abs().max()
    print('   error in grad (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-4, 'error: {}'.format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_initialize_affine_weight(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing initialize_affine_weight with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    _initialize_affine_weight(weight, output_size, input_size,
                                     output_size_coeff, 0,
                                     torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff,
                            dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   column parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    _initialize_affine_weight(weight, output_size, input_size,
                                         input_size_coeff, 1,
                                         torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff,
                            dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   row parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight

class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient accumulation
    fusion in backprop.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce, sequence_parallel):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            world_size = mpu.get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=mpu.initialize.get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias

        if ctx.sequence_parallel:
            world_size = mpu.get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = \
                get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=mpu.initialize.get_tensor_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Convert the tensor shapes to 2D for execution compatibility
        if len(grad_output.shape) > 2:
            grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                           grad_output.shape[2])
            total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
                                           total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=mpu.initialize.get_tensor_model_parallel_group(), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                            group=get_tensor_model_parallel_group(),
                                                            async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # reduce scatter scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1

        if ctx.gradient_accumulation_fusion:
            import fused_dense_cuda
            fused_dense_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None
mpu.LinearWithGradAccumulationAndAsyncCommunication = LinearWithGradAccumulationAndAsyncCommunication
mpu.layers.LinearWithGradAccumulationAndAsyncCommunication = LinearWithGradAccumulationAndAsyncCommunication

def test_column_parallel_linear(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing ColumnParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).npu()
    linear_layer = mpu.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).npu()
    linear_layer.async_tensor_model_parallel_allreduce = False
    loss_weight = torch.randn([batch_size, output_size]).npu()
    # Forward
    input_ = identity_layer()
    output, bias = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.npu()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).npu().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    my_dLdb = torch.split(dLdb, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_row_parallel_linear(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing RowParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).npu()
    linear_layer = mpu.RowParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).npu()
    loss_weight = torch.randn([batch_size, output_size]).npu()
    # Forward
    input_ = identity_layer()
    output, bias = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.npu()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).npu().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff,
                          dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def parallel_self_attention(tensor_model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).npu()
    attention_layer = BertParallelSelfAttention(hidden_size, num_att_heads,
                                                    dropout_prob).npu()
    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).npu()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).npu()
    # Forward
    input_ = identity_layer()
    output = attention_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer


def test_parallel_self_attention(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelSelfAttention with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    dropout_prob = 0.0  # has to be zero
    batch_size = 5
    sequence_length = 13

    rank_1, hideen_size_1, tensor_model_parallel_size_1, loss_1, \
        attention_layer_1, identity_layer_1 = parallel_self_attention(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer = parallel_self_attention(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)
    assert hideen_size_1 == hidden_size

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    my_lin_grad_list = torch.split(
        attention_layer_1.query_key_value.weight.grad,
        hidden_size // tensor_model_parallel_size, 0)[rank::tensor_model_parallel_size]
    my_lin_grad = torch.cat(my_lin_grad_list, dim=0)
    error = my_lin_grad.sub(
        attention_layer.query_key_value.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   weight gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-5

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def parallel_transformer(tensor_model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).npu()
    transformer_layer = BertParallelTransformerLayer(
        hidden_size, intermediate_size, num_att_heads, 0.0, 0.0,
        torch.nn.functional.relu, 1.0e-5).npu()

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).npu()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).npu()
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, tensor_model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-3, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-3, 'error: {}'.format(error)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize_distributed()
    world_size = torch.distributed.get_world_size()
    mpu.destroy_model_parallel()

    print_separator('test initialize affine weight')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_initialize_affine_weight(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test parallel embedding')
        test_parallel_embedding(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test column-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_column_parallel_linear(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test row-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_row_parallel_linear(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test parallel self-attention')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_parallel_self_attention(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test parallel transformer')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_parallel_transformer_layer(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
