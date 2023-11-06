# Megatron-LM

## 1 简介与特性介绍

Megatron-LM 是由 NVIDIA 的应用深度学习研究团队开发的一款功能强大的大型Transformer仓。而本仓库megatron_npu为昇腾基于[Megatron-LM原始仓](https://github.com/NVIDIA/Megatron-LM)所开发的适配仓，已适配特性如下：

- 数据并行（Data parallel）
- 模型并行（Tensor parallel）
- 序列并行（Sequence parallel）
- 流水并行（Pipeline parallel）
- 分布式优化器（Distributed optimizer）

## 2 环境准备
> 基于安全性考虑，建议您以非root的安全账户执行脚本。
### 2.1 Pytorch框架训练环境准备
请参考昇腾官方文档《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。建议您在准备好模型训练环境以后，将umask调整为`027`或以上。


### 2.2 克隆原始仓
  ```shell
  git clone https://github.com/NVIDIA/Megatron-LM.git
  cd Megatron-LM
  git checkout 285068c8108e0e8e6538f54fe27c3ee86c5217a2
  ```
### 2.3 安装 megatron_npu

在Megatron-LM目录下执行：

  ```shell
  git clone https://gitee.com/ascend/Megatron-LM.git megatron_npu_adaptor
  cd megatron_npu_adaptor
  pip install -e .
  ```
> 如需要保存安装megatron_npu的日志，可在pip install命令后面添加参数 `--log <PATH>`，并对您指定的路径`<PATH>`做好权限管控。
### 2.4 安装其他依赖
  ```shell
  pip install -r requirements.txt
  ```
## 3 数据集

### 3.1 获取数据集。
我们以enwiki数据集为例，下载地址为Megatron官方提供，您可以根据自身需求选择数据集训练。

对于步骤1，我们以`./dataset`作为存储数据集的目录，并默认赋予`750`的权限控制，您可以自行选择存放目录，并参考[附录C 文件权限清单](#c-文件权限清单)进行权限设置。

```shell
# 步骤1: 在megatron_npu_adaptor目录下建立数据集目录dataset并进入，您可以选择
mkdir dataset && chmod 750 dataset && cd dataset

# 步骤2: 下载 enwiki 数据集
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
bzip2 -dk enwiki-latest-pages-articles.xml.bz2

# 步骤3: 下载bert-large Vocab表
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt

# 步骤4: 安装 WikiExtractor 对xml格式的数据集进行初步处理
pip3 install wikiextractor
python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml --json
```

### 3.2 处理数据集
在megatron_npu_adaptor根目录执行脚本，对数据集和VOCAB词表进行处理。
   ```shell
   # 退出dataset目录，回到megatron_npu_adaptor根目录
   cd ..
   # 执行数据预处理脚本
   bash ./tests/dataset_preprocess_t5.sh
   ```
### 3.3 数据集目录结构
经过上述步骤处理后，数据集默认放置在megatron_npu_adaptor根目录的`./dataset/en_wiki`下，其目录结构与内容如下所示：

   ```
   ├── ./dataset/en_wiki             
         ├── my-t5_text_sentence.bin
         ├── my-t5_text_sentence.idx
   ```

## 4 训练
以下脚本执行模型的训练与评估，均通过调用torch.distributed.launch函数，函数参数的说明，请您参考[Pytorch官方文档](https://pytorch.org/docs/stable/distributed.html)。
脚本传入launch函数的入参仅作为参考，您可以根据需求调整参数，并建议您参考[关于文件的权限控制](#关于文件的权限控制)，对生成文件（如权重文件checkpoint）进行安全加固。
### 4.1 执行如下前置命令
   ```shell
   cd ./tests_gpt/
   mv pretrain_gpt.py ../../
   ```
### 4.2 运行训练脚本
以单机8卡训练为例：
```shell
bash pretrain_gpt_distributed.sh
```
训练完成后，权重文件默认保存在./checkpoint下，并输出模型训练精度和性能信息。
### 4.3 使能bf16、fp16的大kernel

#### 4.3.1 命令行参数说明

pretrain_gpt_distributed_fp16.sh脚本实际调用并透传以下三个参数到torch.distributed.launch函数调用，具体如下：

|   参数名称    |                           参数说明                           |
| :-----------: | :----------------------------------------------------------: |
|     --pre     | 透传到launch函数的--pre-tockens参数，用于控制flash attention三角的大小 |
|    --next     | 透传到launch函数的--next-tockens参数，用于控制flash attention三角的大小 |
| --shape_order |    透传到launch函数的--shape_order参数，用于输入排布格式     |

#### 4.3.2 执行训练

以单机8卡训练为例：

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
训练完成后，权重文件会默认保存在./checkpoint下，并输出模型训练精度和性能信息。

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
   进入链接：https://catalog.ngc.nvidia.com/orgs/nvidia/models/megatron_lm_345m/files ，选择File Brower，下载release/mp_rank_00/model_optim_rng.pt，放置在`checkpoint_dist_test/mp_rank_00/`，然后执行样例脚本：

   ```shell
   bash test_gpt_distributed_sample.sh
   ```

4. 非大kernel zeroshot评估脚本

   以单机8卡训练为例：
   ```shell
   bash test_gpt_distributed.sh
   ```

5. bf16、fp16的大kernel zeroshot评估脚本

   以单机8卡训练为例，命令行参数说明可参考 4.3.1 命令行参数说明。

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
为了使能Megatron在NPU上运行，我们通过Monkey Patch技术对Megatron原有函数的实现进行替换。用户使用原生Megatron库的接口，运行时执行megatron_npu库中替换的函数实现。megatron_npu不对外暴露接口。

具体被替换实现的内部函数清单详见[附录A 内部函数清单](#a-内部函数清单)

### 6.2 关于Monkey Patch

`Monkey Patch`技术基于Python语言的动态特性，实现了运行时函数的动态替换。

### 6.3 安全加固方案
#### 关于文件的权限控制

- 建议您参考[附录C 文件权限清单](#c-文件权限清单)对各类文件权限进行设计与控制。
- linux系统的umask值建议不低于`027`。

- 建议您务必对模型训练相关文件（如数据集、配置文件、源代码、checkpoint等）做好权限管理，避免文件被恶意篡改、破坏业务进行等风险，比如可以控制为同组/其他用户仅有只读权限。
- 原生megatron以及torch框架执行中所生成的文件权限受到linux系统umask参数影响，如umask设置为`027`，其目录/文件权限默认为`750`/`640`，您可进一步管理权限。
#### 关于命令执行
基于安全性考虑，建议您在执行任何命令时，都尽量使用非root账户执行，遵循权限最小化原则。

#### 关于资源使用

建议您根据自身运行环境资源状况，进行训练配置的设定与数据集的准备，若与资源状况不匹配，比如数据集的size超出内存容量/NPU存储容量等，那么原生的Megatron或Pytorch库的组件会直接退出，并自动释放占用的资源。

#### 关于数据集与index map

第一次执行训练，原生megatron会打印`WARNING: could not find index map files`，并尝试**在数据集目录下帮您创建index map files**，从而能够继续训练。为兼容多用户共享数据集文件以及`index map files`的业务场景，生成的`index map files`权限默认为`644`，存在被其他用户访问的风险，您可以参考[附录C 文件权限清单](#c-文件权限清单)对其进行加固。

#### 关于通信

您作为计算集群的完全控制者，务必注意集群节点间的通信安全，比如做好组网设计并采取相关安全措施。建议在内部网络下部署计算集群，从而避免公网环境下的诸多安全风险。

#### 关于网络端口
megatron_npu不主动开放端口，对于原生Pytorch开放的相关端口，您可以参考其官方文档进行设置。在单机训练的情况下，不建议开放全局端口。具体的通信矩阵可以参考[附录D 通信矩阵](#d-通信矩阵)。

#### 关于算子编译

megatron_npu不涉及C++算子编译。

运行时底层的CANN会缓存算子编译文件，存储在运行目录下的`kernel_meta_*`文件夹内，加快后续训练的运行速度。

#### 关于环境变量

megatron定义了环境变量 `high_precision` 来切换高精度和低精度模式，默认值为False，即为低精度模式，用户可以自行选择。其他环境变量均为CANN等底层依赖引入，可以参考相应官方文档。

#### 关于公网地址

megatron_npu的示例脚本含有公网地址，为公开LICENSE的地址，具体请参考[附录B 公网地址说明](#b-公网地址说明)。

### 6.4 关于卸载

- Pytorch框架训练环境的卸载可以参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes/ptes_00032.html)。

- megatron_npu的卸载只需执行命令：

  > 如需要保存卸载megatron_npu的日志，可在pip uninstall命令后面添加参数 `--log <PATH>`，并对您指定的路径`<PATH>`做好权限管控。
  
  ```python
  pip uninstall megatron_npu
  ```

## 附录

### A-内部函数清单

|                         原生函数位置                         |                接口说明                 |          对应megatron_npu文件          |
| :----------------------------------------------------------: | :-------------------------------------: | :------------------------------------: |
|            megatron.arguments._add_training_args             |             设置命令行参数              |          adaptor_arguments.py          |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward |           并行交叉熵前向计算            |     adaptor_core_cross_entropy.py      |
| megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.backward |           并行交叉熵反向更新            |     adaptor_core_cross_entropy.py      |
| megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward |           并行词嵌入前向计算            |         adaptor_core_layers.py         |
|   megatron.core.tensor_parallel.random._set_cuda_rng_state   |           并行词嵌入反向更新            |    adaptor_core_tensor_parallel.py     |
|       megatron.core.utils._kernel_make_viewless_tensor       |            viewless张量构造             |         adaptor_core_utils.py          |
|       megatron.data.gpt_dataset._build_index_mappings        |         gpt_dataset构建mapping          |      adaptor_data_gpt_dataset.py       |
|               megatron.set_jit_fusion_options                |             设置JIT融合配置             |         adaptor_initialize.py          |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.\__init__ |        MixedFusedLayerNorm初始化        |   adaptor_model_fused_layer_norm.py    |
| megatron.model.fused_layer_norm.MixedFusedLayerNorm.forward  |       MixedFusedLayerNorm前向计算       |   adaptor_model_fused_layer_norm.py    |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available |      FusedScaleMaskSoftmax内部函数      |     adaptor_model_fused_softmax.py     |
| megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax |             前向融合softmax             |     adaptor_model_fused_softmax.py     |
|            megatron.model.module.fp32_to_float16             |               fp32转fp16                |        adaptor_model_module.py         |
|            megatron.model.module.float16_to_fp32             |               fp16转fp32                |        adaptor_model_module.py         |
|       megatron.model.transformer.ParallelMLP.\__init__       |            ParallelMLP初始化            |      adaptor_model_transformer.py      |
|        megatron.model.transformer.ParallelMLP.forward        |           ParallelMLP前向计算           |      adaptor_model_transformer.py      |
|       megatron.model.transformer.CoreAttention.forward       |          CoreAttention前向计算          |      adaptor_model_transformer.py      |
|        megatron.model.transformer.FlashSelfAttention         |         FlashSelfAttention对象          |      adaptor_model_transformer.py      |
|    megatron.model.transformer.ParallelAttention.\__init__    |         ParallelAttention初始化         |      adaptor_model_transformer.py      |
|     megatron.model.transformer.ParallelAttention.forward     |        ParallelAttention前向计算        |      adaptor_model_transformer.py      |
|                 megatron.clip_grad_norm_fp32                 |        fp32下梯度clip与norm操作         |    adaptor_optimizer_clip_grads.py     |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.\__init__ |       DistributedOptimizer初始化        | adaptor_optimizer_distrib_optimizer.py |
| megatron.optimizer.distrib_optimizer.DistributedOptimizer.build_model_and_main_param_groups |          模型与参数组构建构建           | adaptor_optimizer_distrib_optimizer.py |
| megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan |         unscale梯度并检查Nan值          |     adaptor_optimizer_optimizer.py     |
|                   megatron.optimizer.Adam                    |              Adam训练优化               |     adaptor_optimizer_optimizer.py     |
| megatron.optimizer.optimizer.Float16OptimizerWithFloat16Params.\__init__ | Float16OptimizerWithFloat16Params初始化 |     adaptor_optimizer_optimizer.py     |
|           megatron.p2p_communication._communicate            |        p2p_communication内部函数        |      adaptor_p2p_communication.py      |
|               megatron.schedules.backward_step               |            schedule反向计算             |          adaptor_schedules.py          |
|      megatron.schedules.forward_backward_no_pipelining       |              前向后向计算               |          adaptor_schedules.py          |
|         megatron.schedules.deallocate_output_tensor          |           deallocate输出张量            |          adaptor_schedules.py          |

### B-公网地址说明

|      类型      |               开源代码地址                |          文件名          |             公网IP地址/公网URL地址/域名/邮箱地址             |          用途说明           |
| :------------: | :---------------------------------------: | :----------------------: | :----------------------------------------------------------: | :-------------------------: |
|  开源代码引入  | https://github.com/NVIDIA/Megatron-LM.git |      tests/ut/*.py       |          http://www.apache.org/licenses/LICENSE-2.0          | 开源引入LICENSE说明所需地址 |


### C-文件权限清单

您可以根据自身需要，参考此清单对各类文件进行加固:

|           类型            |  linux权限参考值  |                             备注                             |
| :-----------------------: | :---------------: | :----------------------------------------------------------: |
|       文件夹 / 目录       | `750` (rwxr-x---) |      包括checkpoint保存目录、数据集存放目录，安装目录等      |
|        数据集文件         | `640` (rw-r-----) | 这里的数据集为公开数据集，不涉及隐私数据、商业资产等。另外，若需要共享数据集目录/文件，您可酌情调整为`755`/`644`，并注意调整后存在被其他用户（Others）读取的风险 |
|       运行生成文件        | `640` (rw-r-----) |      如checkpoint、数据集预处理npy文件等就属于生成文件       |
|     不可执行程序文件      | `440` (r--r-----) | 一般程序文件不应修改，如果需要进行开发，您可酌情调整为`640`  |
| 程序目录 / 可执行程序文件 | `550` (r-xr-x---) | 一般程序目录/可执行程序不应修改，如果需要进行开发，您可酌情调整为`750` |

### D-通信矩阵

|           源设备            |    源IP    |                            源端口                            |          目的设备           |   目的IP   |                       目的端口（侦听）                       | 协议 |                           端口说明                           |                             备注                             |
| :-------------------------: | :--------: | :----------------------------------------------------------: | :-------------------------: | :--------: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP | 当用户不使用**测试示例脚本**，则默认29500/29400。用户可调用`torch.distributed.launch`函数，通过传入的`--master_port`自由指定1024-65535之间未被占用的端口 | TCP  | 源端口与目的端口均用于收发数据。对于静态分布式场景（backend=static）默认端口为29400；对于动态分布式场景（backend=c10d）中默认端口29500 | megatron_npu本身不开启端口，该通信过程由开源软件Pytorch控制，配置方式可参考其官方文档：https://pytorch.org/docs/stable/distributed.html#launch-utility |
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP | 当使用`pretrain_gpt_distributed*`系列测试示例脚本，脚本对`torch.distributed.launch`传入的`--master_port`为**6000**，用户可以自由指定1024-65535之间未被占用的端口 | TCP  | 原生Pytorch（调用`torchrun`、`torch.distributed.launch`）通信需要，用于收发数据 | 和第一条记录所述为同一端口，这里特别说明**测试示例脚本**对Pytorch开启的master_port默认配置为6000 |
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP | 当使用`test_gpt_distributed*`系列测试示例脚本，脚本对`torch.distributed.launch`传入的`--master_port`为**60035**，用户可以自由指定1024-65535之间未被占用的端口 | TCP  | 原生Pytorch（调用`torchrun`、`torch.distributed.launch`）通信需要，用于收发数据 | 和第一条记录所述为同一端口，这里特别说明**测试示例脚本**对Pytorch开启的master_port默认配置为60035 |
| 运行torch_npu进程的计算设备 | 设备地址IP |                  请参见备注中的CANN官方文档                  | 运行torch_npu进程的计算设备 | 设备地址IP |                  请参见备注中的CANN官方文档                  | TCP  |                  请参见备注中的CANN官方文档                  | 该通信过程完全由HCCL组件控制，端口范围可参考文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha001/ref/envref/envref_07_0065.html CANN通信文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha001/ref/hcclapiref/hcclapi_07_0001.html |

