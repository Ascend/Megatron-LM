# Megatron-LM

-   [概述](概述.md)
-   [准备训练环境](准备训练环境.md)
-   [开始训练](开始训练.md)
-   [训练结果展示](训练结果展示.md)
-   [版本说明](版本说明.md)


# 概述

## 简述

Megatron 是由 NVIDIA 的应用深度学习研究团队开发的一款功能强大的大型Transformer仓。此仓为昇腾基于github原始仓的适配仓，主要实现的特性为tensor-model-parallel和T5-mp2模型。

- 参考实现：

  ```
  url=https://github.com/NVIDIA/Megatron-LM.git
  commit_id=0bb597b42c53355a567aba2a1357cc34b9d99ddd
  ```

- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/Megatron-LM
  code_path=./
  ```
  
- 通过Git获取代码方法如下：

  ```
  git clone {url}       # 克隆仓库的代码
  cd {code_path}        # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
  
- 通过单击“立即下载”，下载源码包。

# 准备训练环境

## 准备环境

- 当前模型支持的固件与驱动、 CANN 以及 PyTorch 如下表所示。

  **表 1**  版本配套表

  | 配套        | 版本                                                         |
  | ---------- | ------------------------------------------------------------ |
  | 固件与驱动   | [22.0.RC3](https://www.hiascend.com/hardware/firmware-drivers?tag=commercial) |
  | CANN       | [6.1.RC1](https://www.hiascend.com/software/cann/commercial?version=6.1.RC1) |
  | PyTorch    | [1.8.1](https://gitee.com/ascend/pytorch/tree/master/)|

- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 克隆原始仓

  ```
  cd Megatron-LM
  git clone https://github.com/NVIDIA/Megatron-LM.git
  # 进入github上拉下来的Megatron-LM
  cd ./Megatron-LM
  git checkout 0bb597b42c53355a567aba2a1357cc34b9d99ddd
  cd -
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
>该数据集的训练过程脚本只作为一种参考示例。


## 获取预训练模型（可选）

- 本模型不涉及

## 测试UT（可选）

```
bash tests/test.sh
```

# 开始训练

## 训练模型

1. 进入解压后的源码包根目录。

   ```
   cd ./${模型文件夹名称} 
   ```

2. 运行训练脚本。

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











