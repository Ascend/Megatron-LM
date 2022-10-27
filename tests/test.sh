source tests/env_npu.sh


python3 -m torch.distributed.launch --nproc_per_node=8 ./tests/ut/test_cross_entropy.py
python3 -m torch.distributed.launch --nproc_per_node=8 ./tests/ut/test_data.py
python3 -m torch.distributed.launch --nproc_per_node=8 ./tests/ut/test_initialize.py
python3 -m torch.distributed.launch --nproc_per_node=8 ./tests/ut/test_layers.py
python3 -m torch.distributed.launch --nproc_per_node=8 ./tests/ut/test_random.py

cd -

