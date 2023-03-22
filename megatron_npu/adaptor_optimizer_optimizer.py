import torch
import megatron


def _unscale_main_grads_and_check_for_nan(self):
    main_grads = self._collect_main_grad_data_for_unscaling()
    self.found_inf.fill_(0.0)
    torch._amp_foreach_non_finite_check_and_unscale_(main_grads, self.found_inf, self.grad_scaler.inv_scale)
    torch.distributed.all_reduce(self.found_inf, op=torch.distributed.ReduceOp.MAX, group=self.get_model_parallel_group())
    torch.distributed.all_reduce(self.found_inf, op=torch.distributed.ReduceOp.MAX, group=mpu.get_data_parallel_group())
    found_inf_flag = (self.found_inf.item() > 0)
    return found_inf_flag


megatron.optimizer.Adam = torch.optim.AdamW
megatron.optimizer.optimizer.MixedPrecisionOptimizer._unscale_main_grads_and_check_for_nan = _unscale_main_grads_and_check_for_nan
