import torch
import torch.nn.functional as F
import megatron
from megatron import get_timers, get_args, core, get_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.model.fused_bias_gelu import bias_gelu_impl


def ParallelMLPForward(self, hidden_states):

    # [s, b, 4hp]
    intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

    if False:
         intermediate_parallel = \
                 bias_gelu_impl(intermediate_parallel, bias_parallel)
    else:
        intermediate_parallel = \
            torch.fast_gelu(intermediate_parallel + bias_parallel)

    # [s, b, h]
    output, output_bias = self.dense_4h_to_h(intermediate_parallel)
    return output, output_bias

def CoreAttentionForward(self, query_layer, key_layer,
            value_layer, attention_mask):

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                   query_layer.size(2),
                   query_layer.size(0),
                   key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2],
                                   output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                               output_size[0] * output_size[1], -1)

    # preallocting input tensor: [b * np, sq, sk]
    matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
        (output_size[0]*output_size[1], output_size[2], output_size[3]),
        query_layer.dtype, "mpu")

    # Raw attention scores. [b * np, sq, sk]
    #matmul_result = torch.baddbmm(
    #    matmul_input_buffer,
    #    query_layer.transpose(0, 1),   # [b * np, sq, hn]
    #    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    #    beta=0.0, alpha=(1.0/self.norm_factor))
    matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.permute(1, 2, 0))
    matmul_result *= 1.0/self.norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    attention_probs = self.scale_mask_softmax(attention_scores,
                                              attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.

    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                   value_layer.size(2),
                   query_layer.size(0),
                   value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                   output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                           output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
        (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


megatron.model.transformer.ParallelMLP.forward = ParallelMLPForward
megatron.model.transformer.CoreAttention.forward = CoreAttentionForward