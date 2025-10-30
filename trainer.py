from typing import Tuple, List, Optional
import os
import heapq
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from torch.optim.swa_utils import AveragedModel
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .functional import (
    resolve_lossfn,
    resolve_optimizer,
    resolve_lr_scheduler,
)
from .config import NetConfig
from .logger import ZeroLogger
from .qc import get_default_unit

HARTREE_TO_EV = 27.211386024367243

class loss2file:
    def __init__(self, loss: float, ptfile: str, epoch: int):
        self.loss = loss
        self.ptfile = ptfile
        self.epoch = epoch

    def __lt__(self, other: "loss2file"):
        return self.loss > other.loss

class AverageMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((1,), device=self.device)
        self.cnt = torch.zeros((1,), dtype=torch.int32, device=self.device)
    
    def update(self, val: float, n: int = 1):
        self.sum += val
        self.cnt += n
    
    def reduce(self) -> float:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg.item()

class EarlyStopping:
    def __init__(
        self, patience: int = None, min_delta: float = 0.0, min_lr: float = 1e-6,
    ):
        self.patience = patience if patience is not None else float("inf")
        self.min_delta = min_delta
        self.min_lr = 1e-6 if min_lr == 0.0 else min_lr
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float, best_loss: float, lr: float):
        if val_loss - best_loss > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        elif lr <= self.min_lr:
            self.stop = True
        else:
            self.counter = 0
        return self.stop

class Trainer:
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: Optional[DistributedSampler],
        log: ZeroLogger,
    ):
        """
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dist_sampler = dist_sampler
        self.log = log

        self.lossfn = resolve_lossfn(config.lossfn, huber_delta=getattr(config, 'huber_delta', 1.0)).to(device)
        self.optimizer = resolve_optimizer(
            optim_type=config.optimizer,
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.max_lr,
            **config.optim_kwargs,
        )
        self.lr_scheduler = resolve_lr_scheduler(
            sched_type=config.lr_scheduler,
            optimizer=self.optimizer,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            max_epochs=config.max_epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=config.warmup_epochs,
            **config.lr_sche_kwargs,
        )
        self.early_stop = EarlyStopping(
            patience=config.early_stop, 
            min_delta=getattr(config, 'min_delta', 0.0),
            min_lr=config.min_lr
        )
        self.ema_model = None
        if config.ema_decay is not None:
            model_to_average = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            ema_model = AveragedModel(
                model_to_average,
                avg_fn=lambda avg_param, param, num_avg: \
                    config.ema_decay * avg_param + (1 - config.ema_decay) * param,
                device=device,
            )
            self.ema_model = ema_model
        self.meter = AverageMeter(device=device)
        self.best_l2fs: List[loss2file] = [
            loss2file(float("inf"), os.path.join(config.save_dir, f"{config.run_name}_{i}.pt"), 0)
            for i in range(config.best_k)
        ]
        
        self.start_epoch = 1
        if config.ckpt_file is not None:
            self._load_params(config.ckpt_file)

    def _load_params(self, ckpt_file: str):
        state = torch.load(ckpt_file, map_location=self.device)
        self.model.module.load_state_dict(state["model"], strict=False)
        if self.config.resume:
            self.optimizer.load_state_dict(state["optimizer"])
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.start_epoch = state["epoch"] + 1
            for l2f in self.best_l2fs:
                if os.path.isfile(l2f.ptfile):
                    pt_state = torch.load(l2f.ptfile, map_location=self.device)
                    l2f.loss = pt_state["loss"] if "loss" in pt_state else float("inf")
        self.log.f.info(f" --- Loaded checkpoint from {ckpt_file}")

    def _save_params(self, model: nn.Module, ckpt_file: str, loss: float = None):
        state = {
            "model": model.state_dict(),
            "epoch": self.epoch,
            "loss": loss,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config.model_hyper_params(),
        }
        torch.save(state, ckpt_file)

    def save_best_k(self, model: nn.Module, curr_loss: float):
        if curr_loss < self.best_l2fs[0].loss:
            l2f = heapq.heappop(self.best_l2fs)
            l2f.loss = curr_loss
            l2f.epoch = self.epoch
            self._save_params(model, l2f.ptfile, l2f.loss)
            heapq.heappush(self.best_l2fs, l2f)

    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
                
            real = data.y - data.base_y if hasattr(data, "base_y") else data.y
            loss = self.lossfn(pred, real)
            
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            if self.ema_model is not None:
                try:
                    self.ema_model.update_parameters(self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model)
                except RuntimeError as e:
                    if "size" in str(e) and "must match" in str(e):
                        print(f"Using adaptive EMA update due to parameter mismatch: {e}")
                        adaptive_ema_update(
                            self.ema_model, 
                            self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model,
                            self.config.ema_decay
                        )
                    else:
                        raise e
            
            if self.config.lr_scheduler != "plateau":
                self.lr_scheduler.step()
            
            if step % self.config.log_step == 0 or step == len(self.train_loader):
                with torch.no_grad():
                    mae_display = F.l1_loss(pred, real).item()
                
                prop_unit = getattr(self.config, 'default_property_unit', 'hartree')

                self.log.f.info(
                    "Epoch: {iepoch:3d} | Step: {step:4d}/{nstep:4d} | LR: {lr:.2e} | Train MAE: {mae:10.7f} {unit}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae=mae_display,
                        unit=prop_unit
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        model_to_validate = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        with torch.no_grad():
            for step, data in enumerate(self.valid_loader, start=1):
                data = data.to(self.device)
                output = model_to_validate(data)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output

                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss_sum_batch = F.l1_loss(pred, real, reduction="sum") 
                self.meter.update(l1loss_sum_batch.item(), real.numel())
                
        mae_hartree = self.meter.reduce()
        
        prop_unit = getattr(self.config, 'default_property_unit', 'hartree')
            
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(
                 "Epoch: {iepoch:3d} | Valid MAE: {mae:10.7f} {unit}".format(
                    iepoch=self.epoch,
                    mae=mae_hartree,
                    unit=prop_unit
                )
            )
        
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(mae_hartree)
        self.early_stop(mae_hartree, self.best_l2fs[0].loss, lr)
        
        self.save_best_k(self.model.module, mae_hartree)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt", loss=mae_hartree)

    def ema_validate(self):
        if self.ema_model is None:
            return float("inf"), float("inf")

        self.ema_model.eval()
        self.meter.reset()

        with torch.no_grad():
            for step, data in enumerate(self.valid_loader, start=1):
                data = data.to(self.device)
                output = self.ema_model(data)
                pred = output
                
                real = data.y - data.base_y if hasattr(data, "base_y") else data.y
                l1loss_sum_batch = F.l1_loss(pred, real, reduction="sum")
                self.meter.update(l1loss_sum_batch.item(), real.numel())
        
        mae_hartree = self.meter.reduce()
        
        prop_unit = getattr(self.config, 'default_property_unit', 'hartree')
            
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(
                "Epoch: {iepoch:3d} | EMA Valid MAE: {mae:10.7f} {unit}".format(
                    iepoch=self.epoch,
                    mae=mae_hartree,
                    unit=prop_unit
                )
            )
        
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(mae_hartree)
        self.early_stop(mae_hartree, self.best_l2fs[0].loss, lr)
        
        self.save_best_k(self.ema_model.module, mae_hartree)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last_ema.pt", loss=mae_hartree)

    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")

        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            
            self.train1epoch()

            self.log.f.info(f"DEBUG: Epoch {iepoch} ended. self.ema_model is {type(self.ema_model)}")
            if self.ema_model is None:
                self.log.f.info("DEBUG: Calling self.validate()")
                self.validate()
            else:
                self.log.f.info("DEBUG: Calling self.ema_validate()")
                self.ema_validate()
            
            if self.early_stop.stop:
                self.log.f.info(f" --- Early Stopping at Epoch {iepoch}")
                break
        
        self.log.f.info(" --- Training Completed")
        self.log.f.info(f" --- Best Valid MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile} at Epoch {self.best_l2fs[-1].epoch}")

class WithForceMeter:
    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self):
        self.sum = torch.zeros((2,), device=self.device)
        self.cnt = torch.zeros((2,), dtype=torch.int32, device=self.device)

    def update(self, energy: float, force: float, n_ene: int, n_frc: int):
        self.sum[0] += energy; self.sum[1] += force
        self.cnt[0] += n_ene; self.cnt[1] += n_frc

    def reduce(self) -> Tuple[float, float]:
        tmp_sum = self.sum.clone()
        tmp_cnt = self.cnt.clone()
        dist.all_reduce(tmp_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tmp_cnt, op=dist.ReduceOp.SUM)
        avg = tmp_sum / tmp_cnt
        return avg[0].item(), avg[1].item()

class GradTrainer(Trainer):
    """
    def __init__(
        self,
        model: nn.parallel.DistributedDataParallel,
        config: NetConfig,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        dist_sampler: DistributedSampler,
        log: ZeroLogger,
    ):
        """
        super().__init__(
            model, config, device, train_loader, valid_loader, dist_sampler, log
        )
        assert config.force_weight <= 1.0
        self.meter = WithForceMeter(self.device)
        self.lossfn_ene = self.lossfn
        self.lossfn_frc = resolve_lossfn(config.lossfn_frc, huber_delta=getattr(config, 'huber_delta', 1.0)).to(device)
        self.loss_w = config.loss_w
        
        if config.ema_decay is not None:
            model_to_average = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            ema_model = AveragedModel(
                model_to_average,
                avg_fn=lambda avg_param, param, num_avg: \
                    config.ema_decay * avg_param + (1 - config.ema_decay) * param,
                device=device,
            )
            self.ema_model = ema_model

    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        for step, data in enumerate(self.train_loader, start=1):
            self.meter.reset()
            data = data.to(self.device)
            data.pos.requires_grad_(True)
            predE, predF = self.model(data)
            realE, realF = data.y, data.force
            if hasattr(data, "base_y") and hasattr(data, "base_force"):
                realE -= data.base_y
                realF -= data.base_force
            lossE = self.lossfn_ene(predE, realE)
            lossF = self.lossfn_frc(predF, realF)
            loss = (1 - self.config.force_weight) * lossE + self.config.force_weight * lossF
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()

            if self.ema_model is not None:
                try:
                    self.ema_model.update_parameters(self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model)
                except RuntimeError as e:
                    if "size" in str(e) and "must match" in str(e):
                        print(f"Using adaptive EMA update due to parameter mismatch: {e}")
                        adaptive_ema_update(
                            self.ema_model, 
                            self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model,
                            self.config.ema_decay
                        )
                    else:
                        raise e

            if self.config.lr_scheduler != "plateau":
                self.lr_scheduler.step()
            
            with torch.no_grad():
                l1lossE = F.l1_loss(predE, realE, reduction="sum")
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())

            if (step % self.config.log_step == 0 or step == len(self.train_loader)):
                maeE_display, maeF_display = self.meter.reduce()
                
                prop_unit = getattr(self.config, 'default_property_unit', 'hartree')
                
                self.log.f.info(
                    "Epoch: {iepoch:3d} | Step: {step:4d}/{nstep:4d} | LR: {lr:.2e} | Train MAE: Energy {maeE:10.7f} {unit} Force {maeF:10.7f} {unit}".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        maeE=maeE_display,
                        maeF=maeF_display,
                        unit=prop_unit
                    )
                )
    
    def validate(self):
        self.model.eval()
        self.meter.reset()
        model_to_validate = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

        with torch.no_grad():
            for step, data in enumerate(self.valid_loader, start=1):
                data = data.to(self.device)
                output_ene, output_frc = model_to_validate(data)
                pred_ene = output_ene
                pred_frc = output_frc
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(pred_ene, realE, reduction="sum") 
                l1lossF = F.l1_loss(pred_frc, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
                
        maeE_display, maeF_display = self.meter.reduce()
        
        prop_unit = getattr(self.config, 'default_property_unit', 'hartree')
            
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(
                "Epoch: {iepoch:3d} | Valid MAE: Energy {maeE:10.7f} {unit}  Force {maeF:10.7f} {unit}".format(
                    iepoch=self.epoch,
                    maeE=maeE_display,
                    maeF=maeF_display,
                    unit=prop_unit
                )
            )
        
        combined_mae = (1 - self.config.force_weight) * maeE_display + self.config.force_weight * maeF_display
        
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(maeE_display)
        
        self.early_stop(maeE_display, self.best_l2fs[0].loss, lr)
        
        self.save_best_k(self.model.module, maeE_display)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt", loss=maeE_display)

    def ema_validate(self):
        if self.ema_model is None:
            return float("inf"), float("inf")

        self.ema_model.eval()
        self.meter.reset()

        with torch.no_grad():
            for step, data in enumerate(self.valid_loader, start=1):
                data = data.to(self.device)
                output_ene, output_frc = self.ema_model(data)
                pred_ene = output_ene
                pred_frc = output_frc
                realE, realF = data.y, data.force
                if hasattr(data, "base_y") and hasattr(data, "base_force"):
                    realE -= data.base_y
                    realF -= data.base_force
                l1lossE = F.l1_loss(pred_ene, realE, reduction="sum")
                l1lossF = F.l1_loss(pred_frc, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
        
        maeE_display, maeF_display = self.meter.reduce()
        
        prop_unit = getattr(self.config, 'default_property_unit', 'hartree')
            
        if self.epoch % self.config.log_epoch == 0:
            self.log.f.info(
                "Epoch: {iepoch:3d} | EMA Valid MAE: Energy {maeE:10.7f} {unit}  Force {maeF:10.7f} {unit}".format(
                    iepoch=self.epoch,
                    maeE=maeE_display,
                    maeF=maeF_display,
                    unit=prop_unit
                )
            )
        
        combined_mae = (1 - self.config.force_weight) * maeE_display + self.config.force_weight * maeF_display
        
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(maeE_display)
        
        self.early_stop(maeE_display, self.best_l2fs[0].loss, lr)
        
        self.save_best_k(self.ema_model.module, maeE_display)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last_ema.pt", loss=maeE_display)

    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")

        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            
            self.train1epoch()

            self.log.f.info(f"DEBUG: Epoch {iepoch} ended. self.ema_model is {type(self.ema_model)}")
            if self.ema_model is None:
                self.log.f.info("DEBUG: Calling self.validate()")
                self.validate()
            else:
                self.log.f.info("DEBUG: Calling self.ema_validate()")
                self.ema_validate()
            
            if self.early_stop.stop:
                self.log.f.info(f" --- Early Stopping at Epoch {iepoch}")
                break
        
        self.log.f.info(" --- Training Completed")
        self.log.f.info(f" --- Best Valid MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile} at Epoch {self.best_l2fs[-1].epoch}")

def adaptive_ema_update(ema_model, current_model, decay):
    """
    try:
        ema_params = dict(ema_model.named_parameters())
        current_params = dict(current_model.named_parameters())
        
        updated_count = 0
        replaced_count = 0
        added_count = 0
        
        for name, current_param in current_params.items():
            if name in ema_params:
                ema_param = ema_params[name]
                
                if ema_param.shape == current_param.shape:
                    try:
                        ema_param.data.mul_(decay).add_(current_param.data, alpha=1 - decay)
                        updated_count += 1
                    except Exception as e:
                        ema_param.data.copy_(current_param.data)
                        replaced_count += 1
                else:
                    try:
                        ema_param.data = current_param.data.clone()
                        replaced_count += 1
                    except Exception as e:
                        try:
                            param_parts = name.split('.')
                            module = ema_model.module if hasattr(ema_model, 'module') else ema_model
                            
                            for part in param_parts[:-1]:
                                if hasattr(module, part):
                                    module = getattr(module, part)
                                else:
                                    break
                            else:
                                param_name = param_parts[-1]
                                new_param = nn.Parameter(current_param.data.clone())
                                setattr(module, param_name, new_param)
                                replaced_count += 1
                        except Exception as e2:
                            print(f"Warning: Failed to update parameter {name}: {e2}")
            else:
                try:
                    param_parts = name.split('.')
                    module = ema_model.module if hasattr(ema_model, 'module') else ema_model
                    
                    for part in param_parts[:-1]:
                        if hasattr(module, part):
                            module = getattr(module, part)
                        else:
                            break
                    else:
                        param_name = param_parts[-1]
                        new_param = nn.Parameter(current_param.data.clone())
                        setattr(module, param_name, new_param)
                        added_count += 1
                except Exception as e:
                    print(f"Warning: Failed to add parameter {name}: {e}")
        
        try:
            ema_buffers = dict(ema_model.named_buffers())
            current_buffers = dict(current_model.named_buffers())
            
            buffer_updated_count = 0
            
            for name, current_buffer in current_buffers.items():
                if name in ema_buffers:
                    ema_buffer = ema_buffers[name]
                    try:
                        if ema_buffer.shape == current_buffer.shape:
                            ema_buffer.data.copy_(current_buffer.data)
                        else:
                            ema_buffer.data = current_buffer.data.clone()
                        buffer_updated_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to update buffer {name}: {e}")
                else:
                    try:
                        buffer_parts = name.split('.')
                        module = ema_model.module if hasattr(ema_model, 'module') else ema_model
                        
                        for part in buffer_parts[:-1]:
                            if hasattr(module, part):
                                module = getattr(module, part)
                            else:
                                break
                        else:
                            buffer_name = buffer_parts[-1]
                            module.register_buffer(buffer_name, current_buffer.clone())
                            buffer_updated_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to add buffer {name}: {e}")
        
        except Exception as e:
            print(f"Warning: Buffer update failed: {e}")
        
        total_changes = replaced_count + added_count
        if total_changes > 0:
            print(f"EMA update: {updated_count} normal, {replaced_count} replaced, {added_count} added")
    
    except Exception as e:
        print(f"Critical error in adaptive_ema_update: {e}")
        print("Falling back to parameter copying...")
        
        try:
            current_state = current_model.state_dict()
            ema_model.load_state_dict(current_state, strict=False)
            print("Fallback: Successfully copied model state")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
