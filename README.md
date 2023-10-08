# Megatron-LM

## 简介

Megatron 是由 NVIDIA 的应用深度学习研究团队开发的一款功能强大的大型Transformer仓。此仓为昇腾基于github原始仓的适配仓，已适配特性如下：

- 数据并行（Data parallel）
- 模型并行（Tensor parallel）
- 序列并行（Sequence parallel）
- 流水并行（Pipeline parallel）
- 分布式优化器（Distributed optimizer）

## 准备环境

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 克隆原始仓
  ```
  git clone https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM
  git checkout 285068c8108e0e8e6538f54fe27c3ee86c5217a2
  ```

- 下载安装 Megatron_npu
  ```
  git clone https://gitee.com/ascend/Megatron-LM.git megatron_npu
  cd megatron_npu
  pip install -e .
  ```

- 安装依赖（根据模型需求，按需添加所需依赖）。
  ```
  pip install -r requirements.txt
  ```

## 准备数据集

1. 获取数据集。

   ```bash ./tests/dataset_preprocess_t5.sh```

2. 数据集目录结构
   将数据集默认放置在```./dataset/en_wiki```下，数据集的目录结构如下所示：

   ```
   ├── ./dataset/en_wiki
         ├── bert-large-uncased-vocab.txt               
         ├── my-t5_text_sentence.bin
         ├── my-t5_text_sentence.idx
   ```

> **说明：**
> 该数据集的训练过程脚本只作为一种**参考**示例。

# 训练

## 预训练

1. 执行如下前置命令
   ```
   cd ./tests_gpt/
   mv pretrain_gpt.py ../../
   ```

2. 运行训练脚本

   该模型支持单机单卡训练和单机8卡训练。

    - 单机8卡训练

      启动8卡训练。

      ```
      bash pretrain_gpt_distributed.sh
      ```

   训练完成后，权重文件保存在./checkpoint下，并输出模型训练精度和性能信息。

3. 使能bf16、fp16的大kernel

   该模型支持单机单卡训练和单机8卡训练。

    - 单机8卡训练

      启动8卡训练。

      ```
      bash pretrain_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=SBH #FP16 flash-attn SBH输入
      bash pretrain_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=BSH #FP16 flash-attn BSH输入
      bash pretrain_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=SBH #FP16 sparse-attn SBH输入
      bash pretrain_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=BSH #FP16 sparse-attn BSH输入
      bash pretrain_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=SBH #BF16 flash-attn SBH输入
      bash pretrain_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=BSH #BF16 flash-attn BSH输入
      bash pretrain_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=SBH #BF16 sparse-attn SBH输入
      bash pretrain_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=BSH #BF16 sparse-attn BSH输入
      ```

   训练完成后，权重文件保存在./checkpoint下，并输出模型训练精度和性能信息。

## LAMBADA Cloze Accuracy

1. 在test_gpt下，执行如下前置命令
   ```
   mv main.py ../../
   mv ../../tasks/zeroshot_gpt ../../
   ```
2. 获取测评数据集
   进入链接：https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl ，下载lambada_test.jsonl，重命名为lambada_test.json，放置在test_gpt下

3. 运行原仓zeroshot评估样例（可跳过）
   进入链接：https://catalog.ngc.nvidia.com/orgs/nvidia/models/megatron_lm_345m/files ，选择File Brower，下载release/mp_rank_00/model_optim_rng.pt，放置在checkpoint_dist_test/mp_rank_00/
   执行样例脚本。
   ```
   bash test_gpt_distributed_sample.sh
   ```

4. 非大kernel zeroshot评估脚本
   - 单机8卡测试
     启动8卡测试。
     ```
     bash test_gpt_distributed.sh
     ```

5. bf16、fp16的大kernel zeroshot评估脚本
   - 单机8卡测试
     启动8卡测试。
     
     ```
     bash test_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=SBH #FP16 flash-attn SBH输入
     bash test_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=BSH #FP16 flash-attn BSH输入
     bash test_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=SBH #FP16 sparse-attn SBH输入
     bash test_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=BSH #FP16 sparse-attn BSH输入
     bash test_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=SBH #BF16 flash-attn SBH输入
     bash test_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=BSH #BF16 flash-attn BSH输入
     bash test_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=SBH #BF16 sparse-attn SBH输入
     bash test_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=BSH #BF16 sparse-attn BSH输入
     ```

# 说明

## 关于接口
为了使能Megatron在NPU上运行，我们通过Monkey Patch技术对Megatron原有函数的实现进行替换，因此megatron_npu与Megatron的原生函数外观保持一致，也不需要对用户暴露其它接口。

具体被替换实现的内部韩式清单详见[附录A 内部函数清单](#A内部函数清单)

### 关于Monkey Patch

`Monkey Patch`技术基于Python语言的动态特性，实现了运行时函数的动态替换，比如megatron_npu可以替换Megatron的部分内部函数实现。

### 关于文件

- 运行命令前，建议用户务必对训练所需文件做好权限控制等安全措施，比如多用户共享数据集的场景下的数据集文件的写权限控制。
- 对于涉及隐私数据、商业资产等敏感文件，建议用户要做好安全防护和权限控制，避免提权等安全风险
- 原生megatron以及torch框架执行中所生成的文件，如参数文件checkpoint，其文件权限默认为644，即仅允许当前执行训练脚本的用户写/读，其他用户仅能读。建议当前执行脚本的用户要对生成文件做好权限控制，避免提权等安全风险。

### 关于网络通信

用户作为计算集群的完全控制者，需要注意集群节点间的通信安全，比如做好组网设计并采取相关安全措施。

### 关于公网地址

megatron_npu的示例脚本与说明文档含有部分公网地址，均为公开数据集、公开代码仓或者公开LICENSE的地址，其中对于示例脚本等用户不能直接感知的公网网址，特地于[附录B 公网地址说明](#B公网地址说明)中列出

## 变更

2022.08.26：首次发布
2023.06.27：新增GPT3

## 已知问题

**当前发行版本中存在的问题描述。**

无。

# 附录

## A内部函数清单

|                         原生函数位置                         |                接口说明                 |          对应megatron_npu文件          | 备注 |
| :----------------------------------------------------------: | :-------------------------------------: | :------------------------------------: | :--: |
|            megatron.arguments._add_training_args             |             设置命令行参数              |          adaptor_arguments.py          |      |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward |           并行交叉熵前向计算            |     adaptor_core_cross_entropy.py      |      |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.backward |           并行交叉熵反向更新            |     adaptor_core_cross_entropy.py      |      |
| megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward |           并行词嵌入前向计算            |         adaptor_core_layers.py         |      |
|   megatron.core.tensor_parallel.random._set_cuda_rng_state   |           并行词嵌入反向更新            |    adaptor_core_tensor_parallel.py     |      |
|       megatron.core.utils._kernel_make_viewless_tensor       |                                         |         adaptor_core_utils.py          |      |
|       megatron.data.gpt_dataset._build_index_mappings        |                                         |      adaptor_data_gpt_dataset.py       |      |
|               megatron.set_jit_fusion_options                |                                         |         adaptor_initialize.py          |      |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.\__init__ |        MixedFusedLayerNorm初始化        |   adaptor_model_fused_layer_norm.py    |      |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward  |                                         |   adaptor_model_fused_layer_norm.py    |      |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available |                                         |     adaptor_model_fused_softmax.py     |      |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax |                                         |     adaptor_model_fused_softmax.py     |      |
|            megatron.model.module.fp32_to_float16             |               fp32转fp16                |        adaptor_model_module.py         |      |
|            megatron.model.module.float16_to_fp32             |               fp16转fp32                |        adaptor_model_module.py         |      |
|       megatron.model.transformer.ParallelMLP.\__init__       |            ParallelMLP初始化            |      adaptor_model_transformer.py      |      |
|        megatron.model.transformer.ParallelMLP.forward        |                                         |      adaptor_model_transformer.py      |      |
|       megatron.model.transformer.CoreAttention.forward       |                                         |      adaptor_model_transformer.py      |      |
|        megatron.model.transformer.FlashSelfAttention         |                                         |      adaptor_model_transformer.py      |      |
|    megatron.model.transformer.ParallelAttention.\__init__    |         ParallelAttention初始化         |      adaptor_model_transformer.py      |      |
|     megatron.model.transformer.ParallelAttention.forward     |                                         |      adaptor_model_transformer.py      |      |
|                 megatron.clip_grad_norm_fp32                 |                                         |    adaptor_optimizer_clip_grads.py     |      |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.\__init__ |       DistributedOptimizer初始化        | adaptor_optimizer_distrib_optimizer.py |      |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups |                                         | adaptor_optimizer_distrib_optimizer.py |      |
| megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan |                                         |     adaptor_optimizer_optimizer.py     |      |
|                   megatron.optimizer.Adam                    |              Adam训练优化               |     adaptor_optimizer_optimizer.py     |      |
| megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.\__init__ | Float16OptimizerWithFloat16Params初始化 |     adaptor_optimizer_optimizer.py     |      |
|           megatron.p2p_communication._communicate            |                                         |      adaptor_p2p_communication.py      |      |
|               megatron.schedules.backward_step               |                                         |          adaptor_schedules.py          |      |
|      megatron.schedules.forward_backward_no_pipelining       |                                         |          adaptor_schedules.py          |      |
|         megatron.schedules.deallocate_output_tensor          |                                         |          adaptor_schedules.py          |      |

## B公网地址说明

|      类型      |               开源代码地址                |          文件名          |             公网IP地址/公网URL地址/域名/邮箱地址             |          用途说明           |
| :------------: | :---------------------------------------: | :----------------------: | :----------------------------------------------------------: | :-------------------------: |
|  开源代码引入  | https://github.com/NVIDIA/Megatron-LM.git |      tests/ut/*.py       |          http://www.apache.org/licenses/LICENSE-2.0          | 开源引入LICENSE说明所需地址 |
| 公开数据集下载 |                                           | dataset_preprocess_t5.sh | https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 |      下载enwiki数据集       |
| 公开数据集下载 |                                           | dataset_preprocess_t5.sh | https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt |         下载词汇表          |







