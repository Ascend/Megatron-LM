import torch
import numbers
import megatron
from megatron.model.fused_layer_norm import MixedFusedLayerNorm, HAVE_PERSIST_LAYER_NORM
from megatron.core.utils import make_viewless_tensor


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


megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__ = MixedFusedLayerNormInit
megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward = MixedFusedLayerNormForward
