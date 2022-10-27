# Mgeatron-LM

## ENV
- github https://github.com/NVIDIA/Megatron-LM.git
- commit_id 0bb597b42c53355a567aba2a1357cc34b9d99ddd

## UT test

- script: ```bash test/test.sh```

- ut_list: (megatron/mpu/tests)
    - [x] test_cross_entropy.py
    - [x] test_data.py
    - [x] test_initialize.py
    - [x] test_layers.py
    - [ ] test_random.py
    
- note
    - get_rng_state not ok now, [issue link](https://gitee.com/ascend/pytorch-develop/issues/I50ZH1?from=project-issue)
    - set seed on npu is not ok now, [issue link](https://gitee.com/ascend/pytorch-develop/issues/I50ZLR?from=project-issue)

## model support plan

### T5 

#### func 

1. [] pretrain_t5.sh
    - ```bash test/pretrain_t5.sh``` 
2. [ ] pretrain_t5_distributed_with_mp.sh
    - ```bash test/pretrain_t5_distributed_with_mp.sh``` 
3. [ ] pretrain_t5_xxB.sh
4. [ ] pretrain_t5_xxxB.sh


### GPT-3

#### func 

1. [ ] pretrain_gpt.sh
2. [ ] pretrain_gpt_distributed.sh
3. [ ] pretrain_gpt_distributed_with_mp.sh
4. [ ] pretrain_gpt3_175B.sh

