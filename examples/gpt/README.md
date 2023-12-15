# GPT3-110B

##  环境准备
GPT3-110B 训练的硬件配置如下:
| 硬件  | 设备  |       配置       |
| :---: | :---: | :--------------: |
|  NPU  |  A+X  | 16 x Ascend NPUs |

GPT3-110B 训练的软件配置如下:

|  软件  |  配置  |
| :----: | :----: |
| python | 3.7.5  |
| torch  | 1.11.0 |

### 环境搭建

请参考《[Megatron-LM 环境准备](https://gitee.com/ascend/Megatron-LM/tree/master#2-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)》。

### 数据集
请参考《[Megatron-LM 数据集](https://gitee.com/ascend/Megatron-LM/tree/master#3-%E6%95%B0%E6%8D%AE%E9%9B%86)》。

###  目录结构

经过上述步骤处理后，当前目录结构与内容如下所示：

```
├── Megatron-LM
│   ├── megatron_npu_adaptor
│          ├── dataset
│          ├── examples
│          ├── megatron_npu
│          ├── test_gpt
│          ├── ......
|   ├── pretrain_gpt.py
│   ├── ......
```
## 训练
### 1. 执行如下前置命令

```shell
cd Megatron-LM
mv pretrain_gpt.py pretrain_gpt_gpu.py
cp ./megatron_npu_adaptor/tests_gpt/pretrain_gpt.py ./
cp ./megatron_npu_adaptor/tests_gpt/env_npu.sh ./
cp ./megatron_npu_adaptor/tests_gpt/gpt2-merges.txt ./
cp ./megatron_npu_adaptor/tests_gpt/gpt2-vocab.json ./
mkdir logs && chmod 750 logs
mkdir checkpoint && chmod 750 checkpoint
```
### 2. 运行训练脚本
以4x16集群为例：
```bash
bash megatron_npu_adaptor/examples/gpt/pretrain_gpt3_110B_4x16.sh
```
> 运行脚本前，需要将脚本中`IPs`变量修改为相应的4机集群IP。
> 启动脚本时，需要在所有机器上启动该脚本。

### 3. 训练结果
GPT3-110B 在 **昇腾芯片** 上的性能：
| 设备 | 模型      | 集群规模 | 单步迭代时间 (s/step) | MFU  |
| ---- | --------- | -------- | --------------------- | ---- |
| NPUs | GPT3-110B | 4x16     | 50.61                 | 0.44 |

## 注意事项
* 用户可以根据自身需要，参考《[Megatron-LM 文件权限清单](https://gitee.com/ascend/Megatron-LM#c-%E6%96%87%E4%BB%B6%E6%9D%83%E9%99%90%E6%B8%85%E5%8D%95)》，对各类文件进行安全加固。