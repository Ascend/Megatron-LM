import math
import torch
import torch.nn.functional as F
import megatron
from megatron import get_timers, get_args, core, get_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model.transformer import _args_to_kwargs
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.transformer import ParallelAttention, CoreAttention
import torch_npu


try:
    from einops import rearrange
except ImportError:
    rearrange = None

def ParallelMLPInit(self, init_method, output_layer_init_method):
    super(ParallelMLP, self).__init__()
    args = get_args()
    self.is_x_model = args.is_x_model
    
    # Project to 4h.
    self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
        args.hidden_size,
        args.ffn_hidden_size,
        gather_output=False,
        init_method=init_method,
        skip_bias_add=True,
        async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
        **_args_to_kwargs())
    
    self.bias_gelu_fusion = args.bias_gelu_fusion
    self.activation_func = F.gelu
    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    
    # Project back to h.
    if self.is_x_model:
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size // 2,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())
    else:
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
                args.ffn_hidden_size,
                args.hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                skip_bias_add=True,
                **_args_to_kwargs())
                
def ParallelMLPForward(self, hidden_states):

    # [s, b, 4hp]
    intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

    if self.is_x_model:
        x = intermediate_parallel + bias_parallel
        x, gates = x.chunk(2, dim=-1)
        intermediate_parallel = x * F.gelu(gates)
    else:
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


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None, pre_tockens=None, next_tockens=None, shape_order='SBH'):
        super().__init__()
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.pre_tockens = pre_tockens
        self.next_tockens = next_tockens
        self.shape_order = shape_order


    def forward(self, q, k, v, n, attention_mask):

        attention_mask = attention_mask.to(q.dtype)

        scale = 1.0/math.sqrt(q.size(2)/n) if self.softmax_scale is None else self.softmax_scale

        output = torch_npu.npu_flash_attention( \
            q, k, v, n, self.shape_order, \
            pse=None, \
            padding_mask=None, \
            atten_mask=attention_mask, \
            scale=scale, \
            pre_tockens=self.pre_tockens, \
            next_tockens=self.next_tockens, \
            keep_prob=1-self.dropout_p, \
            )[0]

        return output



def ParallelAttentionInit(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.sequence_parallel = args.sequence_parallel
        self.shape_order = args.shape_order
        
        self.use_flash_attn = args.use_flash_attn
        if self.use_flash_attn:
        
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())


            self.key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout,
                pre_tockens=args.pre_tockens, next_tockens=args.next_tockens,
                shape_order=args.shape_order
            )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())


def ParallelAttentionForward(self, hidden_states, attention_mask,
            encoder_output=None, inference_params=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_len
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size)
            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================

    if self.attention_type == AttnType.self_attn:
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
         key_layer,
         value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
         value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
        batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
        batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
                    :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
                      :sequence_end, batch_start:batch_end, ...]

    # ==================================
    # core attention computation
    # ==================================

    if not self.use_flash_attn:
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:

        hidden_head_num = query_layer.size(2)
        if self.shape_order == 'BSH':
            q, k, v = [rearrange(x, 's b h d -> b s (h d)').contiguous()
                for x in (query_layer, key_layer, value_layer)]
        elif self.shape_order == 'SBH':
            q, k, v = [rearrange(x, 's b h d -> s b (h d)').contiguous()
                for x in (query_layer, key_layer, value_layer)]
        else:
            raise ImportError('flash attention shape order must be SBH or BSH, please add args shape-order')
            
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)
        else:
            context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)
        
        if self.shape_order == 'BSH':
            context_layer = torch.tensor(1.0).to(context_layer.dtype).npu() * context_layer
            context_layer = rearrange(context_layer, 'b s D -> s b D').contiguous()

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias

megatron.model.transformer.ParallelMLP.__init__ = ParallelMLPInit
megatron.model.transformer.ParallelMLP.forward = ParallelMLPForward
megatron.model.transformer.CoreAttention.forward = CoreAttentionForward
megatron.model.transformer.FlashSelfAttention = FlashSelfAttention
megatron.model.transformer.ParallelAttention.__init__ = ParallelAttentionInit
megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward
