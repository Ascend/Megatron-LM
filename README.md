# Megatron-LM

## 简介

Megatron 是由 NVIDIA 的应用深度学习研究团队开发的一款功能强大的大型Transformer仓。此仓为昇腾基于github原始仓的适配仓，已适配特性如下：

- 数据并行（Data parallel）
- 模型并行（Tensor parallel）
- 序列并行（Sequence parallel）
- 流水并行（Pipeline parallel）
- 分布式优化器（Distributed optimizer）

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

| 配套 | 版本 |
| ----- | ----- |
| 固件与驱动   | [22.0.RC3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
| CANN    | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=6.1.RC1) |
| PyTorch | [1.11](https://gitee.com/ascend/pytorch/tree/master/) |

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
  git clone https://gitee.com/ascend/Megatron-LM.git -b dev megatron_npu
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
   将数据集默认放置在```./dataset/en_wiki/preprocess/```下，数据集的目录结构如下所示：

   ```
   ├── ./dataset/en_wiki/preprocess/
         ├── bert-large-uncased-vocab.txt               
         ├── my-t5_text_sentence.bin
         ├── my-t5_text_sentence.idx
   ```

> **说明：**
> 该数据集的训练过程脚本只作为一种参考示例。

## 获取预训练模型（可选）

- 本模型不涉及

## 测试UT（可选）

```
bash tests/test.sh
```

# 开始训练

## 训练模型

1. 需要在模型执行前导入 megatron_npu 包。
   ```
   import megatron_npu
   ```

2. 进入解压后的源码包根目录。

   ```
   cd ./${模型文件夹名称} 
   ```

3. 运行训练脚本。

   该模型支持单机单卡训练和单机8卡训练。

    - 单机8卡训练

      启动8卡训练。

      ```
      bash ./tests/train_full_8p.sh   
      ```

   训练完成后，权重文件保存在./checkpoint下，并输出模型训练精度和性能信息。

# 训练结果展示

**表 2**  训练结果展示表

| NAME     | PPL    | samples/s | Steps     |
| -------  | -----  |----------:| ------    |
| 8p-竞品A  | 8.688  |       232 | 100000    |
| 8p-NPU   | 8.701  |       100 | 100000    |
| 32p-NPU  | 6.319  |       393 | 100000    |

备注：一定要有竞品和NPU。

# 版本说明

## 变更

2022.08.26：首次发布

## 已知问题

**_当前发行版本中存在的问题描述。_**

无。











