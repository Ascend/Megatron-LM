import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import bugfix

from pretrain_t5 import pretrain, train_valid_test_datasets_provider, model_provider, forward_step
from megatron.model import T5Model, ModelType

option = {}
option["ACL_OP_COMPILER_CACHE_MODE"] = "enable"
option["ACL_OP_COMPILER_CACHE_DIR"] = "./cache"
print("option:",option)
torch.npu.set_option(option)

pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_and_decoder,
         forward_step, args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
