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
> 该数据集的训练过程脚本只作为一种参考示例。

# 开始训练

## 训练模型

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



# 版本说明

## 接口声明
Megatron_npu以Monkey Patching技术替换Megatron原有函数实现，函数外观与Megatron保持一致，属于内部函数。因此在用户层面而言，并未对外提供接口。

## 变更

2022.08.26：首次发布
2023.06.27：新增GPT3

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











