# Megatron-LM

## 1 简介与特性介绍

Megatron 是由 NVIDIA 的应用深度学习研究团队开发的一款功能强大的大型Transformer仓。此仓为昇腾基于github原始仓的适配仓，已适配特性如下：

- 数据并行（Data parallel）
- 模型并行（Tensor parallel）
- 序列并行（Sequence parallel）
- 流水并行（Pipeline parallel）
- 分布式优化器（Distributed optimizer）

## 2 环境准备
> 建议您以非root的安全账户执行脚本，以避免安全风险。
### 2.1 Pytorch框架训练环境准备
请参考昇腾官方文档《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。


### 2.2 克隆原始仓
  ```shell
  git clone https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM
  git checkout 285068c8108e0e8e6538f54fe27c3ee86c5217a2
  ```

### 2.3 下载安装 Megatron_npu
  ```shell
  git clone https://gitee.com/ascend/Megatron-LM.git megatron_npu
  cd megatron_npu
  pip install -e .
  ```
### 2.4 安装其他依赖
> 根据具体场景需求，您可以**按需**添加所需依赖，并注意依赖的版本控制。

  ```shell
  pip install -r requirements.txt
  ```
## 3 数据集

### 3.1 获取数据集。
这里以enwiki数据集为例，下载地址为Megatron官方提供，您可以根据自身需求选择数据集。

```shell
# Step 1: 下载 enwiki 数据集
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -dk enwiki-latest-pages-articles.xml.bz2

# Step 2: 下载bert-large Vocab表
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt

# Step 3: 安装 WikiExtractor 对xml格式的数据集进行初步处理
pip3 install wikiextractor
python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml --json
```

### 3.2 处理数据集
执行脚本，对数据集和VOCAB词表进行处理。
   ```shell
   bash ./tests/dataset_preprocess_t5.sh
   ```
### 3.3 数据集目录结构
   经过上述步骤处理后，数据集默认放置在```./dataset/en_wiki```下，其目录结构与内容如下所示：

   ```
   ├── ./dataset/en_wiki
         ├── bert-large-uncased-vocab.txt               
         ├── my-t5_text_sentence.bin
         ├── my-t5_text_sentence.idx
   ```



## 4 训练

### 4.1 执行如下前置命令
   ```shell
   cd ./tests_gpt/
   mv pretrain_gpt.py ../../
   ```
### 4.2 运行训练脚本
- 单机单卡训练和单机8卡训练。
    ```shell
    bash pretrain_gpt_distributed.sh
    ```
    训练完成后，权重文件保存在./checkpoint下，并输出模型训练精度和性能信息。

### 4.3 使能bf16、fp16的大kernel

- 单机单卡训练和单机8卡训练。
  
  ```shell
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

## 5 评估
### LAMBADA Cloze Accuracy

1. 在test_gpt下，执行如下前置命令
   ```shell
   mv main.py ../../
   mv ../../tasks/zeroshot_gpt ../../
   ```
2. 获取测评数据集
   进入链接：https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl ，下载lambada_test.jsonl，重命名为lambada_test.json，放置在test_gpt下

3. 运行原仓zeroshot评估样例（可跳过）
   进入链接：https://catalog.ngc.nvidia.com/orgs/nvidia/models/megatron_lm_345m/files ，选择File Brower，下载release/mp_rank_00/model_optim_rng.pt，放置在checkpoint_dist_test/mp_rank_00/
   执行样例脚本。
   ```shell
   bash test_gpt_distributed_sample.sh
   ```

4. 非大kernel zeroshot评估脚本
   - 单机8卡测试
     启动8卡测试。
     ```shell
     bash test_gpt_distributed.sh
     ```

5. bf16、fp16的大kernel zeroshot评估脚本
   - 单机8卡测试
     启动8卡测试。
     
     ```shell
     bash test_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=SBH #FP16 flash-attn SBH输入
     bash test_gpt_distributed_fp16.sh --pre=65536 --next=65536 --shape_order=BSH #FP16 flash-attn BSH输入
     bash test_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=SBH #FP16 sparse-attn SBH输入
     bash test_gpt_distributed_fp16.sh --pre=2048 --next=0 --shape_order=BSH #FP16 sparse-attn BSH输入
     bash test_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=SBH #BF16 flash-attn SBH输入
     bash test_gpt_distributed_bf16.sh --pre=65536 --next=65536 --shape_order=BSH #BF16 flash-attn BSH输入
     bash test_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=SBH #BF16 sparse-attn SBH输入
     bash test_gpt_distributed_bf16.sh --pre=2048 --next=0 --shape_order=BSH #BF16 sparse-attn BSH输入
     ```

## 6 说明

### 6.1 关于接口
为了使能Megatron在NPU上运行，我们通过Monkey Patch技术对Megatron原有函数的实现进行替换，因此megatron_npu与Megatron的原生函数外观保持一致，也不需要对用户暴露其它接口。

具体被替换实现的内部函数清单详见[附录A 内部函数清单](#a-内部函数清单)

### 6.2 关于Monkey Patch

`Monkey Patch`技术基于Python语言的动态特性，实现了运行时函数的动态替换，比如megatron_npu可以替换Megatron的部分内部函数实现。

### 6.3 安全加固方案
#### 关于文件的权限控制
- 运行命令前，建议您务必对训练所需文件做好权限控制等安全措施，比如多用户共享数据集的场景下的数据集文件的写权限控制。
- 对于涉及隐私数据、商业资产等敏感文件，建议用户要做好安全防护和权限控制，避免提权等安全风险
- 原生megatron以及torch框架执行中所生成的文件，如参数文件checkpoint，其文件权限默认为`640`，文件夹默认权限为`750`，即写权限只有当前执行训练脚本的用户拥有。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，避免提权等安全风险。
- 建议用户参考[附录C 文件权限清单](#c-文件权限清单)对各类文件权限进行设计与控制。另外，umask的值建议不低于`027`。
#### 关于命令执行
无论是环境准备还是训练等涉及命令执行的操作，建议用户使用非root账户执行，避免可能的安全风险

#### 关于网络通信

您作为计算集群的完全控制者，务必注意集群节点间的通信安全，比如做好组网设计并采取相关安全措施。建议在内部网络下部署计算集群，从而避免公网环境下的诸多安全风险。

#### 关于公网地址

megatron_npu的示例脚本与说明文档含有部分公网地址，均为公开数据集、公开代码仓或者公开LICENSE的地址，其中对于示例脚本等用户不能直接感知的公网网址，请参考[附录B 公网地址说明](#b-公网地址说明)。

### 6.4 已知问题
可参考当前Megatron发行版本中存在的问题描述。

## 附录

### A-内部函数清单

|                         原生函数位置                         |                接口说明                 |          对应megatron_npu文件          | 备注 |
| :----------------------------------------------------------: | :-------------------------------------: | :------------------------------------: | :--: |
|            megatron.arguments._add_training_args             |             设置命令行参数              |          adaptor_arguments.py          |      |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward |           并行交叉熵前向计算            |     adaptor_core_cross_entropy.py      |      |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.backward |           并行交叉熵反向更新            |     adaptor_core_cross_entropy.py      |      |
| megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward |           并行词嵌入前向计算            |         adaptor_core_layers.py         |      |
|   megatron.core.tensor_parallel.random._set_cuda_rng_state   |           并行词嵌入反向更新            |    adaptor_core_tensor_parallel.py     |      |
|       megatron.core.utils._kernel_make_viewless_tensor       |                                         |         adaptor_core_utils.py          |      |
|       megatron.data.gpt_dataset._build_index_mappings        |         gpt_dataset构建mapping          |      adaptor_data_gpt_dataset.py       |      |
|               megatron.set_jit_fusion_options                |             设置JIT融合配置             |         adaptor_initialize.py          |      |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.\__init__ |        MixedFusedLayerNorm初始化        |   adaptor_model_fused_layer_norm.py    |      |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward  |       MixedFusedLayerNorm前向计算       |   adaptor_model_fused_layer_norm.py    |      |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available |      FusedScaleMaskSoftmax内部函数      |     adaptor_model_fused_softmax.py     |      |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax |             前向融合softmax             |     adaptor_model_fused_softmax.py     |      |
|            megatron.model.module.fp32_to_float16             |               fp32转fp16                |        adaptor_model_module.py         |      |
|            megatron.model.module.float16_to_fp32             |               fp16转fp32                |        adaptor_model_module.py         |      |
|       megatron.model.transformer.ParallelMLP.\__init__       |            ParallelMLP初始化            |      adaptor_model_transformer.py      |      |
|        megatron.model.transformer.ParallelMLP.forward        |           ParallelMLP前向计算           |      adaptor_model_transformer.py      |      |
|       megatron.model.transformer.CoreAttention.forward       |          CoreAttention前向计算          |      adaptor_model_transformer.py      |      |
|        megatron.model.transformer.FlashSelfAttention         |         FlashSelfAttention对象          |      adaptor_model_transformer.py      |      |
|    megatron.model.transformer.ParallelAttention.\__init__    |         ParallelAttention初始化         |      adaptor_model_transformer.py      |      |
|     megatron.model.transformer.ParallelAttention.forward     |        ParallelAttention前向计算        |      adaptor_model_transformer.py      |      |
|                 megatron.clip_grad_norm_fp32                 |        fp32下梯度clip与norm操作         |    adaptor_optimizer_clip_grads.py     |      |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.\__init__ |       DistributedOptimizer初始化        | adaptor_optimizer_distrib_optimizer.py |      |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups |          模型与参数组构建构建           | adaptor_optimizer_distrib_optimizer.py |      |
| megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan |         unscale梯度并检查Nan值          |     adaptor_optimizer_optimizer.py     |      |
|                   megatron.optimizer.Adam                    |              Adam训练优化               |     adaptor_optimizer_optimizer.py     |      |
| megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.\__init__ | Float16OptimizerWithFloat16Params初始化 |     adaptor_optimizer_optimizer.py     |      |
|           megatron.p2p_communication._communicate            |        p2p_communication内部函数        |      adaptor_p2p_communication.py      |      |
|               megatron.schedules.backward_step               |            schedule反向计算             |          adaptor_schedules.py          |      |
|      megatron.schedules.forward_backward_no_pipelining       |              前向后向计算               |          adaptor_schedules.py          |      |
|         megatron.schedules.deallocate_output_tensor          |                                         |          adaptor_schedules.py          |      |

### B-公网地址说明

|      类型      |               开源代码地址                |          文件名          |             公网IP地址/公网URL地址/域名/邮箱地址             |          用途说明           |
| :------------: | :---------------------------------------: | :----------------------: | :----------------------------------------------------------: | :-------------------------: |
|  开源代码引入  | https://github.com/NVIDIA/Megatron-LM.git |      tests/ut/*.py       |          http://www.apache.org/licenses/LICENSE-2.0          | 开源引入LICENSE说明所需地址 |


### C-文件权限清单

您可以根据自身需要，参考此清单对各类文件进行加固:

|      类型      | linux权限参考值 |                       备注                       |
| :------------: | :-------------: | :----------------------------------------------: |
| 文件夹 / 目录  | 750 (rwxr-x---) |               对于共享目录可为755                |
|   数据集文件   | 640 (rw-r-----) |            对于共享数据集文件可为644             |
| checkpoint文件 | 640 (rw-r-----) |                                                  |
|    程序文件    | 440 (r--r-----) | 除非开发调试场景，正常运行时程序文件不应再次修改 |
|   可执行脚本   | 750 (rwxr-x---) |                                                  |







