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
from .normalizer import Normalizer, NormalizerManager
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
        monitor_energy: bool = True, monitor_force: bool = True, monitor_combined: bool = True,
        relative_threshold: float = 0.01
    ):
        self.patience = patience if patience is not None else float("inf")
        self.min_delta = min_delta
        self.min_lr = 1e-6 if min_lr == 0.0 else min_lr
        self.relative_threshold = relative_threshold
        self.counter = 0
        self.stop = False
        self.monitor_energy = monitor_energy
        self.monitor_force = monitor_force
        self.monitor_combined = monitor_combined
        self.best_energy_loss = float("inf")
        self.best_force_loss = float("inf")
        self.best_combined_loss = float("inf")
        self.energy_counter = 0
        self.force_counter = 0
        self.combined_counter = 0
    def __call__(self, val_loss: float, best_loss: float, lr: float,
                 energy_loss: float = None, force_loss: float = None):
        if lr <= self.min_lr:
            self.stop = True
            return self.stop
        if self.monitor_combined:
            if self.best_combined_loss > 0:
                relative_improvement = (self.best_combined_loss - val_loss) / self.best_combined_loss
            else:
                relative_improvement = 0.0
            if (val_loss < self.best_combined_loss and
                (relative_improvement > self.relative_threshold or
                 (self.best_combined_loss - val_loss) > self.min_delta)):
                self.best_combined_loss = val_loss
                self.combined_counter = 0
            else:
                self.combined_counter += 1
        if self.monitor_energy and energy_loss is not None:
            if energy_loss < self.best_energy_loss:
                self.best_energy_loss = energy_loss
                self.energy_counter = 0
            else:
                self.energy_counter += 1
        if self.monitor_force and force_loss is not None:
            if force_loss < self.best_force_loss:
                self.best_force_loss = force_loss
                self.force_counter = 0
            else:
                self.force_counter += 1
        if self.monitor_combined and self.combined_counter >= self.patience:
            self.stop = True
        return self.stop
class Trainer:
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
            params=self.model.parameters(),
            lr=config.max_lr,
            weight_decay=getattr(config, 'weight_decay', 1e-6),
            **getattr(config, 'optim_kwargs', {})
        )
        self.lr_scheduler = resolve_lr_scheduler(
            sched_type=config.lr_scheduler,
            optimizer=self.optimizer,
            max_lr=config.max_lr,
            min_lr=config.min_lr,
            max_epochs=config.max_epochs,
            steps_per_epoch=len(self.train_loader),
            warmup_epochs=config.warmup_epochs,
            **getattr(config, 'lr_sche_kwargs', {})
        )
        self.early_stop = EarlyStopping(
            patience=config.early_stop,
            min_delta=getattr(config, 'min_delta', 0.0),
            min_lr=getattr(config, 'min_lr', 1e-6),
            monitor_energy=True,
            monitor_force=True,
            monitor_combined=True,
            relative_threshold=getattr(config, 'early_stop_relative_threshold', 0.01)
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
            self.log.f.info(f"EMA model initialized with decay={config.ema_decay}")
        else:
            self.log.f.info("EMA model not initialized (ema_decay is None)")
        self.meter = AverageMeter(device=device)
        self.best_l2fs: List[loss2file] = [
            loss2file(float("inf"), os.path.join(config.save_dir, f"{config.run_name}_{i}.pt"), 0)
            for i in range(config.best_k)
        ]
        self.start_epoch = 1
        if config.ckpt_file is not None:
            self.log.f.info(f"Loading checkpoint from: {config.ckpt_file}")
            self._load_params(config.ckpt_file)
            self._sync_parameters_after_loading()
            model_module = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            nan_params_found = False
            for name, param in model_module.named_parameters():
                if torch.isnan(param).any():
                    self.log.f.warning(f"Parameter {name} contains NaN after loading checkpoint")
                    nan_params_found = True
                    if 'out.energy_mlp' in name:
                        self.log.f.warning(f"Re-loading {name} from checkpoint")
                        ckpt_state = torch.load(config.ckpt_file, map_location=self.device)
                        if name in ckpt_state['model']:
                            param.data.copy_(ckpt_state['model'][name])
                            self.log.f.info(f"Successfully re-loaded {name}")
            if not nan_params_found:
                self.log.f.info("All parameters loaded successfully without NaN")
            else:
                self.log.f.warning("Some parameters contained NaN and were re-loaded")
            self.log.f.info("=== Detailed parameter statistics ===")
            for name, param in model_module.named_parameters():
                if torch.isnan(param).any():
                    self.log.f.warning(f"Parameter {name} contains NaN")
                elif torch.isinf(param).any():
                    self.log.f.warning(f"Parameter {name} contains Inf")
                else:
                    self.log.f.info(f"Parameter {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, min={param.min().item():.6f}, max={param.max().item():.6f}")
    def _load_params(self, ckpt_file: str):
        state = torch.load(ckpt_file, map_location=self.device)
        missing_keys, unexpected_keys = self.model.module.load_state_dict(state["model"], strict=False)
        if missing_keys:
            self.log.f.warning(f"Missing keys during loading: {missing_keys}")
            model_state = self.model.module.state_dict()
            ckpt_state = state["model"]
            energy_mlp_keys = [k for k in ckpt_state.keys() if 'out.energy_mlp' in k]
            if energy_mlp_keys:
                self.log.f.info(f"Found {len(energy_mlp_keys)} energy_mlp keys in checkpoint")
                loaded_count = 0
                for key in energy_mlp_keys:
                    if key in model_state:
                        if model_state[key].shape == ckpt_state[key].shape:
                            model_state[key].copy_(ckpt_state[key])
                            self.log.f.info(f"Successfully loaded {key} with shape {ckpt_state[key].shape}")
                            loaded_count += 1
                        else:
                            self.log.f.warning(f"Shape mismatch for {key}: checkpoint {ckpt_state[key].shape} vs model {model_state[key].shape}")
                    else:
                        self.log.f.warning(f"Key {key} not found in current model")
                self.log.f.info(f"Loaded {loaded_count}/{len(energy_mlp_keys)} energy_mlp parameters")
            self.model.module.load_state_dict(model_state, strict=False)
            self.log.f.info("Checking for NaN in loaded parameters...")
            nan_found = False
            for name, param in self.model.module.named_parameters():
                if torch.isnan(param).any():
                    self.log.f.warning(f"Parameter {name} contains NaN after loading")
                    nan_found = True
                    if 'out.energy_mlp' in name and name in ckpt_state:
                        self.log.f.info(f"Re-loading {name} from checkpoint")
                        param.data.copy_(ckpt_state[name])
                        if torch.isnan(param).any():
                            self.log.f.error(f"Parameter {name} still contains NaN after re-loading")
                        else:
                            self.log.f.info(f"Successfully re-loaded {name}")
            if not nan_found:
                self.log.f.info("All parameters loaded successfully without NaN")
        if unexpected_keys:
            self.log.f.warning(f"Unexpected keys during loading: {unexpected_keys}")
        if self.config.resume:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                self.log.f.info(" --- Successfully loaded optimizer state")
            except ValueError as e:
                if "different number of parameter groups" in str(e):
                    self.log.f.warning(" --- Optimizer parameter groups mismatch, skipping optimizer state loading")
                    self.log.f.warning(f" --- Error: {e}")
                else:
                    raise e
            try:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
                self.log.f.info(" --- Successfully loaded learning rate scheduler state")
            except Exception as e:
                self.log.f.warning(f" --- Failed to load learning rate scheduler state: {e}")
                self.log.f.warning(" --- Will continue with fresh scheduler state")
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
    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'ema_model_state_dict': self.ema_model.module.state_dict() if self.ema_model and hasattr(self.ema_model, 'module') else (self.ema_model.state_dict() if self.ema_model else None),
            'best_loss': self.best_l2fs[0].loss if self.best_l2fs else float('inf'),
            'config': self.config,
            'early_stop_state': self.early_stop.__dict__ if hasattr(self, 'early_stop') else None,
        }
        checkpoint_file = f"{self.config.save_dir}/{self.config.run_name}_checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_file)
        self.log.f.info(f" --- Saved checkpoint: {checkpoint_file}")
        self._cleanup_old_checkpoints()
    def _cleanup_old_checkpoints(self):
        import glob
        import os
        pattern = f"{self.config.save_dir}/{self.config.run_name}_checkpoint_epoch_*.pt"
        checkpoint_files = glob.glob(pattern)
        if len(checkpoint_files) > 3:
            checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
            for old_file in checkpoint_files[:-3]:
                try:
                    os.remove(old_file)
                    self.log.f.info(f" --- Removed old checkpoint: {old_file}")
                except Exception as e:
                    self.log.f.info(f" --- Failed to remove old checkpoint {old_file}: {e}")
    def _sync_parameters_after_loading(self):
        if not dist.is_initialized():
            return
        self.log.f.info("Synchronizing parameters after checkpoint loading...")
        model_module = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        for name, param in model_module.named_parameters():
            if param.requires_grad:
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
        self.log.f.info("Parameter synchronization completed")
    def load_checkpoint(self, checkpoint_file):
        if not os.path.exists(checkpoint_file):
            self.log.f.info(f" --- Checkpoint file not found: {checkpoint_file}")
            return False
        try:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['lr_scheduler_state_dict'] and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if checkpoint['ema_model_state_dict'] and self.ema_model:
                if hasattr(self.ema_model, 'module'):
                    self.ema_model.module.load_state_dict(checkpoint['ema_model_state_dict'])
                else:
                    self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.epoch = checkpoint['epoch']
            if checkpoint['early_stop_state'] and hasattr(self, 'early_stop'):
                self.early_stop.__dict__.update(checkpoint['early_stop_state'])
            self.log.f.info(f" --- Loaded checkpoint from epoch {checkpoint['epoch']}: {checkpoint_file}")
            return True
        except Exception as e:
            self.log.f.info(f" --- Failed to load checkpoint {checkpoint_file}: {e}")
            return False
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
            nan_gradients_found = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    self.log.f.warning(f"Gradient {name} contains NaN, zeroing it")
                    param.grad.data.zero_()
                    nan_gradients_found = True
            if nan_gradients_found:
                self.log.f.warning("NaN gradients detected and zeroed, skipping this step")
                self.optimizer.zero_grad()
                continue
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if self.ema_model is not None:
                try:
                    if hasattr(self.ema_model, 'trainable_param_names'):
                        self.ema_model.update_parameters(self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model)
                    else:
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
                        print(f"EMA update failed with error: {e}")
                        pass
            if step % self.config.log_step == 0 or step == len(self.train_loader):
                with torch.no_grad():
                    mae_display = F.l1_loss(pred, real).item()
                    mae_physical = mae_display * self.config.target_std
                self.log.f.info(
                    "Epoch: {iepoch:3d} | Step: {step:4d}/{nstep:4d} | LR: {lr:.2e} | Train MAE: {mae:.4f} hartree".format(
                        iepoch=self.epoch,
                        step=step,
                        nstep=len(self.train_loader),
                        lr=self.optimizer.param_groups[0]["lr"],
                        mae=mae_physical
                    )
                )
    def validate(self):
        self.model.eval()
        self.meter.reset()
        model_to_validate = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        total_relative_error = 0.0
        total_relative_error_count = 0
        for step, data in enumerate(self.valid_loader, start=1):
            data = data.to(self.device)
            if getattr(self.config, 'output_mode', 'grad') == 'conservative':
                with torch.enable_grad():
                    data.pos.requires_grad_(True)
                    output_ene, output_frc = model_to_validate(data)
                    pred_ene = output_ene
                    pred_frc = output_frc
            else:
                with torch.no_grad():
                    output_ene, output_frc = model_to_validate(data)
                    pred_ene = output_ene
                    pred_frc = output_frc
            realE, realF = data.y, data.force
            if hasattr(data, "base_y") and hasattr(data, "base_force"):
                realE -= data.base_y
                realF -= data.base_force
            with torch.no_grad():
                l1lossE = F.l1_loss(pred_ene, realE, reduction="sum")
                l1lossF = F.l1_loss(pred_frc, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
                if self.config.lossfn.startswith('relative'):
                    relative_errors = torch.abs(pred_ene - realE) / (torch.abs(realE) + 1e-8)
                    total_relative_error += torch.sum(relative_errors).item()
                    total_relative_error_count += relative_errors.numel()
        maeE_norm, maeF_norm = self.meter.reduce()
        maeE_physical = maeE_norm * self.config.target_std
        maeF_physical = maeF_norm * self.config.grad_target_std
        if self.epoch % self.config.log_epoch == 0:
            if self.config.lossfn.startswith('relative'):
                avg_relative_error = total_relative_error / total_relative_error_count if total_relative_error_count > 0 else 0.0
                self.log.f.info(
                    "Epoch: {iepoch:3d} | Valid MAE: Energy {maeE_rel:.4f} (rel)  Force {maeF:.4f} hartree/bohr".format(
                        iepoch=self.epoch,
                        maeE_rel=avg_relative_error,
                        maeF=maeF_physical
                    )
                )
            else:
                self.log.f.info(
                    "Epoch: {iepoch:3d} | Valid MAE: Energy {maeE:.8f} Force {maeF:.8f}".format(
                        iepoch=self.epoch,
                        maeE=maeE_norm,
                        maeF=maeF_norm
                    )
                )
        energy_coeff = self.config.energy_coefficient
        force_coeff = self.config.force_coefficient
        combined_loss_norm = energy_coeff * maeE_norm + force_coeff * maeF_norm
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(combined_loss_norm)
        self.early_stop(combined_loss_norm, self.best_l2fs[0].loss, lr,
                       energy_loss=maeE_norm, force_loss=maeF_norm)
        if self.epoch % 10 == 0:
            if self.early_stop.best_combined_loss > 0:
                relative_improvement = (self.early_stop.best_combined_loss - combined_loss_norm) / self.early_stop.best_combined_loss
            else:
                relative_improvement = 0.0
            self.log.f.info(
                f"EMA Early Stop Status - Epoch {self.epoch}: "
                f"Combined Counter: {self.early_stop.combined_counter}/{self.early_stop.patience}, "
                f"Energy Counter: {self.early_stop.energy_counter}/{self.early_stop.patience}, "
                f"Force Counter: {self.early_stop.force_counter}/{self.early_stop.patience}, "
                f"Best Loss: {self.early_stop.best_combined_loss:.2e}, "
                f"Current Loss: {combined_loss_norm:.2e}, "
                f"Relative Improvement: {relative_improvement:.2%}"
            )
        if self.epoch % 10 == 0:
            if self.early_stop.best_combined_loss > 0:
                relative_improvement = (self.early_stop.best_combined_loss - combined_loss_norm) / self.early_stop.best_combined_loss
            else:
                relative_improvement = 0.0
            self.log.f.info(
                f"Early Stop Status - Epoch {self.epoch}: "
                f"Combined Counter: {self.early_stop.combined_counter}/{self.early_stop.patience}, "
                f"Energy Counter: {self.early_stop.energy_counter}/{self.early_stop.patience}, "
                f"Force Counter: {self.early_stop.force_counter}/{self.early_stop.patience}, "
                f"Best Loss: {self.early_stop.best_combined_loss:.2e}, "
                f"Current Loss: {combined_loss_norm:.2e}, "
                f"Relative Improvement: {relative_improvement:.2%}"
            )
        self.save_best_k(self.model.module, combined_loss_norm)
        self._save_params(self.model.module, f"{self.config.save_dir}/{self.config.run_name}_last.pt", loss=combined_loss_norm)
    def ema_validate(self):
        if self.ema_model is None:
            return float("inf"), float("inf")
        self.ema_model.eval()
        self.meter.reset()
        total_relative_error = 0.0
        total_relative_error_count = 0
        for step, data in enumerate(self.valid_loader, start=1):
            data = data.to(self.device)
            if getattr(self.config, 'output_mode', 'grad') == 'conservative':
                with torch.enable_grad():
                    data.pos.requires_grad_(True)
                    output_ene, output_frc = self.ema_model(data)
                    pred_ene = output_ene
                    pred_frc = output_frc
            else:
                with torch.no_grad():
                    output_ene, output_frc = self.ema_model(data)
                    pred_ene = output_ene
                    pred_frc = output_frc
            realE, realF = data.y, data.force
            if hasattr(data, "base_y") and hasattr(data, "base_force"):
                realE -= data.base_y
                realF -= data.base_force
            with torch.no_grad():
                l1lossE = F.l1_loss(pred_ene, realE, reduction="sum")
                l1lossF = F.l1_loss(pred_frc, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
                if self.config.lossfn.startswith('relative'):
                    relative_errors = torch.abs(pred_ene - realE) / (torch.abs(realE) + 1e-8)
                    total_relative_error += torch.sum(relative_errors).item()
                    total_relative_error_count += relative_errors.numel()
        maeE_norm, maeF_norm = self.meter.reduce()
        maeE_physical = maeE_norm * self.config.target_std
        maeF_physical = maeF_norm * self.config.grad_target_std
        if self.epoch % self.config.log_epoch == 0:
            if self.config.lossfn.startswith('relative'):
                avg_relative_error = total_relative_error / total_relative_error_count if total_relative_error_count > 0 else 0.0
                self.log.f.info(
                    "Epoch: {iepoch:3d} | EMA Valid MAE: Energy {maeE_rel:.4f} (rel)  Force {maeF:.4f} hartree/bohr".format(
                        iepoch=self.epoch,
                        maeE_rel=avg_relative_error,
                        maeF=maeF_physical
                    )
                )
            else:
                self.log.f.info(
                    "Epoch: {iepoch:3d} | EMA Valid MAE: Energy {maeE:.8f} Force {maeF:.8f}".format(
                        iepoch=self.epoch,
                        maeE=maeE_norm,
                        maeF=maeF_norm
                    )
                )
        energy_coeff = self.config.energy_coefficient
        force_coeff = self.config.force_coefficient
        combined_loss_norm = energy_coeff * maeE_norm + force_coeff * maeF_norm
        lr = self.optimizer.param_groups[0]["lr"]
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler.step(combined_loss_norm)
        self.early_stop(combined_loss_norm, self.best_l2fs[0].loss, lr,
                       energy_loss=maeE_norm, force_loss=maeF_norm)
        self.save_best_k(self.ema_model.module, combined_loss_norm)
        self._save_params(self.ema_model.module, f"{self.config.save_dir}/{self.config.run_name}_last_ema.pt", loss=combined_loss_norm)
        if hasattr(self.config, 'save_checkpoint_every') and self.config.save_checkpoint_every > 0:
            if self.epoch % self.config.save_checkpoint_every == 0:
                self.save_checkpoint()
    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")
        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            self.current_epoch = iepoch
            self.train1epoch()
            if self.lr_scheduler is not None:
                if hasattr(self, 'is_differential_scheduler') and self.is_differential_scheduler:
                    if hasattr(self.lr_scheduler, 'step_epoch'):
                        self.lr_scheduler.step_epoch(self.epoch)
                        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                        if hasattr(self.lr_scheduler, 'warmup_epochs'):
                            if self.epoch < self.lr_scheduler.warmup_epochs:
                                phase = f"WARMUP ({self.epoch}/{self.lr_scheduler.warmup_epochs})"
                            else:
                                cosine_epoch = self.epoch - self.lr_scheduler.warmup_epochs
                                phase = f"COSINE ({cosine_epoch}/{self.lr_scheduler.cosine_T_max})"
                        else:
                            phase = "UNKNOWN"
                        self.log.f.info(f"Epoch {self.epoch} 学习率更新 [{phase}]: {current_lrs}")
                    else:
                        self.lr_scheduler.step()
                else:
                    self.lr_scheduler.step()
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
        self.log.f.info(f" --- Best Valid Force MAE: {self.best_l2fs[-1].loss:.5f}")
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
        super().__init__(
            model, config, device, train_loader, valid_loader, dist_sampler, log
        )
        if hasattr(config, 'energy_lr') and hasattr(config, 'force_lr') and config.energy_lr is not None and config.force_lr is not None:
            self.log.f.info("检测到差分学习率配置，设置差分优化器...")
            self._setup_differential_optimizer()
        else:
            self.log.f.info("未检测到差分学习率配置，使用统一学习率...")
            if self.config.optimizer == "adamW":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.max_lr,
                    **self.config.optim_kwargs
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            self.lr_scheduler = resolve_lr_scheduler(
                sched_type=self.config.lr_scheduler,
                optimizer=self.optimizer,
                max_lr=self.config.max_lr,
                min_lr=self.config.min_lr,
                max_epochs=self.config.max_epochs,
                steps_per_epoch=len(self.train_loader),
                warmup_epochs=self.config.warmup_epochs,
                **self.config.lr_sche_kwargs,
            )
            self.log.f.info(f"统一学习率: {self.config.max_lr}")
            self.log.f.info(f"权重衰减: {self.config.optim_kwargs.get('weight_decay', 'N/A')}")
        self.meter = WithForceMeter(self.device)
        self.lossfn_ene = resolve_lossfn(
            config.lossfn,
            huber_delta=getattr(config, 'huber_delta', 1.0),
            eps=getattr(config, 'relative_error_eps', 1e-8),
            delta=getattr(config, 'relative_error_delta', 0.1)
        ).to(device)
        self.lossfn_frc = resolve_lossfn(
            config.lossfn_frc,
            huber_delta=getattr(config, 'huber_delta', 1.0)
        ).to(device)
        self.loss_w = config.loss_w
        self.normalizers = NormalizerManager(device=device)
        if config.normalize_labels:
            if config.target_mean is not None and config.target_std is not None:
                self.normalizers.add_normalizer(
                    "target",
                    Normalizer(
                        mean=config.target_mean,
                        std=config.target_std,
                        device=device
                    )
                )
            if config.grad_target_mean is not None and config.grad_target_std is not None:
                self.normalizers.add_normalizer(
                    "grad_target",
                    Normalizer(
                        mean=config.grad_target_mean,
                        std=config.grad_target_std,
                        device=device
                    )
                )
        if config.ema_decay is not None and self.ema_model is None:
            model_to_average = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            ema_model = AveragedModel(
                model_to_average,
                avg_fn=lambda avg_param, param, num_avg: \
                    config.ema_decay * avg_param + (1 - config.ema_decay) * param,
                device=device,
            )
            self.ema_model = ema_model
            self.log.f.info(f"EMA model initialized for GradTrainer with decay={config.ema_decay}")
        elif config.ema_decay is not None and self.ema_model is not None:
            self.log.f.info(f"EMA model already initialized in parent class with decay={config.ema_decay}")
        if getattr(config, 'output_mode', 'grad') == 'conservative':
            self.log.f.info(f"Conservative mode: 使用配置文件中的权重设置")
            self.log.f.info(f"energy_coefficient: {getattr(config, 'energy_coefficient', 'N/A')}")
            self.log.f.info(f"force_coefficient: {getattr(config, 'force_coefficient', 'N/A')}")
            self.log.f.info(f"理论依据：传统保守力训练方法，权重完全由配置文件控制")
            self.log.f.info(f"学习率策略：兼容第一阶段参数组结构，使用统一学习率")
        if config.lossfn.startswith('relative'):
            self.log.f.info(f"损失函数策略：能量使用相对误差({config.lossfn})，力使用绝对误差({config.lossfn_frc})")
            self.log.f.info(f"相对误差参数：eps={getattr(config, 'relative_error_eps', 1e-8)}, delta={getattr(config, 'relative_error_delta', 0.1)}")
        else:
            self.log.f.info(f"损失函数策略：能量使用绝对误差({config.lossfn})，力使用绝对误差({config.lossfn_frc})")
            if config.lossfn == "smoothl1":
                self.log.f.info(f"SmoothL1损失参数：beta={getattr(config, 'huber_delta', 1.0)}")
            elif config.lossfn == "huber":
                self.log.f.info(f"Huber损失参数：delta={getattr(config, 'huber_delta', 1.0)}")
        self.current_epoch = 0
        self.dynamic_force_coeff = None
        self._static_balance_logged = False
        self._gradient_explosion_count = 0
        self._max_gradient_explosions = 100
        self._spherical_grad_clip = 2.0
        self._global_grad_clip = 1.0
    def train1epoch(self):
        self.model.train()
        self.dist_sampler.set_epoch(self.epoch)
        total_relative_error = 0.0
        total_relative_error_count = 0
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
            energy_coeff = getattr(self.config, 'energy_coefficient', 0.1)
            force_coeff = getattr(self.config, 'force_coefficient', 1.0)
            if getattr(self.config, 'output_mode', 'grad') == 'conservative':
                energy_coeff = getattr(self.config, 'energy_coefficient', 0.1)
                force_coeff = getattr(self.config, 'force_coefficient', 1.0)
                loss = energy_coeff * lossE + force_coeff * lossF
            elif getattr(self.config, 'output_mode', 'grad') == 'body_order_energy':
                energy_coeff = getattr(self.config, 'energy_coefficient', 0.1)
                force_coeff = getattr(self.config, 'force_coefficient', 1.0)
                base_loss = energy_coeff * lossE + force_coeff * lossF
                smoothness_loss = torch.tensor(0.0, device=lossE.device)
                if hasattr(self.model.module.out, 'compute_smoothness_loss'):
                    smoothness_loss = self.model.module.out.compute_smoothness_loss(predE, data.pos)
                smoothness_weight = getattr(self.config, 'smoothness_weight', 0.01)
                loss = base_loss + smoothness_weight * smoothness_loss
            else:
                energy_coeff = getattr(self.config, 'energy_coefficient', 0.1)
                force_coeff = getattr(self.config, 'force_coefficient', 1.0)
                if getattr(self.config, 'adaptive_balance', False):
                    current_epoch = self.current_epoch
                    total_epochs = self.config.max_epochs
                    progress = current_epoch / total_epochs
                    if progress < 0.3:
                        dynamic_force_coeff = force_coeff * 0.5
                    elif progress < 0.7:
                        dynamic_force_coeff = force_coeff * 1.0
                    else:
                        dynamic_force_coeff = force_coeff * 1.5
                    loss = energy_coeff * lossE + dynamic_force_coeff * lossF
                    self.dynamic_force_coeff = dynamic_force_coeff
                    if self.current_epoch % 5 == 0:
                        self.log.f.info(f"Dynamic balance: epoch {self.current_epoch}, progress {progress:.2f}, force_coeff {dynamic_force_coeff:.2f}")
                else:
                    loss = energy_coeff * lossE + force_coeff * lossF
                    if not hasattr(self, '_static_balance_logged'):
                        self.log.f.info(f"Static balance: energy_coeff={energy_coeff:.1f}, force_coeff={force_coeff:.1f}")
                        self._static_balance_logged = True
            self.optimizer.zero_grad()
            loss.backward()
            nan_gradients_found = False
            inf_gradients_found = False
            large_gradients_found = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        self.log.f.warning(f"NaN gradient found in {name}")
                        nan_gradients_found = True
                    if torch.isinf(param.grad).any():
                        self.log.f.warning(f"Inf gradient found in {name}")
                        inf_gradients_found = True
                    if torch.abs(param.grad).max() > 100:
                        self.log.f.warning(f"Large gradient found in {name}: max={torch.abs(param.grad).max().item():.2f}")
                        large_gradients_found = True
            if nan_gradients_found or inf_gradients_found:
                self.log.f.warning("NaN or Inf gradients detected, zeroing gradients and skipping step")
                self.optimizer.zero_grad()
                continue
            if large_gradients_found:
                self.log.f.warning("Large gradients detected, applying moderate clipping")
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-10, 10)
            if self.config.grad_clip is not None:
                if (hasattr(self.config, 'energy_grad_clip') and hasattr(self.config, 'force_grad_clip')
                    and self.config.energy_grad_clip is not None and self.config.force_grad_clip is not None):
                    model_module = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
                    energy_params = list(model_module.out.energy_mlp.parameters())
                    if energy_params and self.config.energy_grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(energy_params, self.config.energy_grad_clip)
                    if hasattr(model_module.out, 'force_mlp'):
                        force_params = list(model_module.out.force_mlp.parameters())
                        if force_params and self.config.force_grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(force_params, self.config.force_grad_clip)
                    other_params = []
                    for name, param in model_module.named_parameters():
                        if not name.startswith('out.'):
                            other_params.append(param)
                    if other_params and self.config.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(other_params, self.config.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
            self.optimizer.step()
            if self.ema_model is not None:
                try:
                    if hasattr(self.ema_model, 'trainable_param_names'):
                        self.ema_model.update_parameters(self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model)
                    else:
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
                        print(f"EMA update failed with error: {e}")
                        pass
            with torch.no_grad():
                l1lossE = F.l1_loss(predE, realE, reduction="sum")
                l1lossF = F.l1_loss(predF, realF, reduction="sum")
                self.meter.update(l1lossE.item(), l1lossF.item(), realE.numel(), realF.numel())
                if self.config.lossfn.startswith('relative'):
                    relative_errors = torch.abs(predE - realE) / (torch.abs(realE) + 1e-8)
                    total_relative_error += torch.sum(relative_errors).item()
                    total_relative_error_count += relative_errors.numel()
                else:
                    total_relative_error = 0.0
                    total_relative_error_count = 0
            if (step % self.config.log_step == 0 or step == len(self.train_loader)):
                maeE_norm, maeF_norm = self.meter.reduce()
                maeE_physical = maeE_norm * self.config.target_std
                maeF_physical = maeF_norm * self.config.grad_target_std
                if len(self.optimizer.param_groups) > 1:
                    lr_info = f"LR: {self.optimizer.param_groups[0]['lr']:.2e} (shared) {self.optimizer.param_groups[1]['lr']:.2e} (energy) {self.optimizer.param_groups[2]['lr']:.2e} (force)"
                else:
                    lr_info = f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                if self.config.lossfn.startswith('relative'):
                    avg_relative_error = total_relative_error / total_relative_error_count if total_relative_error_count > 0 else 0.0
                    self.log.f.info(
                        "Epoch: {iepoch:3d} | Step: {step:4d}/{nstep:4d} | {lr_info} | Train MAE: Energy {maeE_rel:.4f} (rel) Force {maeF:.4f} hartree/bohr".format(
                            iepoch=self.epoch,
                            step=step,
                            nstep=len(self.train_loader),
                            lr_info=lr_info,
                            maeE_rel=avg_relative_error,
                            maeF=maeF_physical
                        )
                    )
                else:
                    self.log.f.info(
                        "Epoch: {iepoch:3d} | Step: {step:4d}/{nstep:4d} | {lr_info} | Train MAE: Energy {maeE:.8f} Force {maeF:.8f}".format(
                            iepoch=self.epoch,
                            step=step,
                            nstep=len(self.train_loader),
                            lr_info=lr_info,
                            maeE=maeE_norm,
                            maeF=maeF_norm
                        )
                    )
    def start(self):
        prop_unit, len_unit = get_default_unit()
        self.log.f.info(" --- Start training")
        self.log.f.info(f" --- Task Name: {self.config.run_name}")
        self.log.f.info(f" --- Property: {self.config.label_name} --- Unit: {prop_unit} {len_unit}")
        for iepoch in range(self.start_epoch, self.config.max_epochs + 1):
            self.epoch = iepoch
            self.current_epoch = iepoch
            self.train1epoch()
            if self.lr_scheduler is not None:
                if hasattr(self, 'is_differential_scheduler') and self.is_differential_scheduler:
                    if hasattr(self.lr_scheduler, 'step_epoch'):
                        self.lr_scheduler.step_epoch(self.epoch)
                        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
                        if hasattr(self.lr_scheduler, 'warmup_epochs'):
                            if self.epoch < self.lr_scheduler.warmup_epochs:
                                phase = f"WARMUP ({self.epoch}/{self.lr_scheduler.warmup_epochs})"
                            else:
                                cosine_epoch = self.epoch - self.lr_scheduler.warmup_epochs
                                phase = f"COSINE ({cosine_epoch}/{self.lr_scheduler.cosine_T_max})"
                        else:
                            phase = "UNKNOWN"
                        self.log.f.info(f"Epoch {self.epoch} 学习率更新 [{phase}]: {current_lrs}")
                    else:
                        self.lr_scheduler.step()
                else:
                    self.lr_scheduler.step()
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
        self.log.f.info(f" --- Best Valid Force MAE: {self.best_l2fs[-1].loss:.5f}")
        self.log.f.info(f" --- Best Checkpoint: {self.best_l2fs[-1].ptfile} at Epoch {self.best_l2fs[-1].epoch}")
    def _setup_differential_optimizer(self):
        if hasattr(self.config, 'energy_lr') and hasattr(self.config, 'force_lr') and self.config.energy_lr is not None and self.config.force_lr is not None:
            if getattr(self.config, 'output_mode', 'grad') == 'conservative':
                self.log.f.info("使用统一学习率模式（保守力训练）")
            else:
                self.log.f.info("使用差分学习率模式（直接力预测）")
            param_groups = []
            shared_params = []
            for name, param in self.model.module.named_parameters():
                if not name.startswith('out.'):
                    shared_params.append(param)
            if shared_params:
                param_groups.append({
                    "params": shared_params,
                    "lr": self.config.min_lr,
                    "weight_decay": self.config.optim_kwargs.get('weight_decay', 0.0),
                    "name": "shared_layers"
                })
                shared_param_count = sum(p.numel() for p in shared_params)
                self.log.f.info(f"GNN主体参数数量: {shared_param_count}")
                self.log.f.info(f"GNN主体学习率: {self.config.max_lr:.6f} (最保守，warmup后)")
            if hasattr(self.model.module, 'out') and hasattr(self.model.module.out, 'energy_mlp'):
                energy_params = list(self.model.module.out.energy_mlp.parameters())
                if energy_params:
                    param_groups.append({
                        "params": energy_params,
                        "lr": self.config.min_lr,
                        "weight_decay": getattr(self.config, 'energy_weight_decay', self.config.optim_kwargs.get('weight_decay', 0.0)),
                        "name": "energy_head"
                    })
                    energy_param_count = sum(p.numel() for p in energy_params)
                    self.log.f.info(f"能量头参数数量: {energy_param_count}")
                    self.log.f.info(f"能量头学习率: {self.config.energy_lr:.6f} (中等，warmup后)")
            if hasattr(self.model.module, 'out') and hasattr(self.model.module.out, 'force_mlp'):
                force_params = list(self.model.module.out.force_mlp.parameters())
                if force_params:
                    param_groups.append({
                        "params": force_params,
                        "lr": self.config.min_lr,
                        "weight_decay": getattr(self.config, 'force_weight_decay', self.config.optim_kwargs.get('weight_decay', 0.0)),
                        "name": "force_head"
                    })
                    force_param_count = sum(p.numel() for p in force_params)
                    self.log.f.info(f"力头参数数量: {force_param_count}")
                    self.log.f.info(f"力头学习率: {self.config.force_lr:.6f} (最高，warmup后)")
            if hasattr(self.model.module, 'out'):
                other_out_params = []
                for name, param in self.model.module.out.named_parameters():
                    if not name.startswith('energy_mlp.') and not name.startswith('force_mlp.'):
                        other_out_params.append(param)
                if other_out_params:
                    param_groups.append({
                        "params": other_out_params,
                        "lr": self.config.min_lr,
                        "weight_decay": self.config.optim_kwargs.get('weight_decay', 0.0),
                        "name": "output_other"
                    })
                    other_param_count = sum(p.numel() for p in other_out_params)
                    self.log.f.info(f"输出层其他参数数量: {other_param_count}")
                    self.log.f.info(f"输出层其他学习率: {self.config.max_lr:.6f} (基础，warmup后)")
            if not param_groups:
                self.log.f.warning("未找到有效的参数组，回退到统一学习率模式")
                param_groups = [{
                    "params": self.model.parameters(),
                    "lr": self.config.max_lr,
                    "weight_decay": self.config.optim_kwargs.get('weight_decay', 0.0),
                    "name": "all_params"
                }]
            if self.config.optimizer == "adamW":
                self.optimizer = torch.optim.AdamW(param_groups, **self.config.optim_kwargs)
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            self.log.f.info("=== 差分学习率参数组设置 ===")
            self.log.f.info("设计原则: 主体要稳，两翼要快，力翼为先")
            for i, group in enumerate(param_groups):
                param_count = sum(p.numel() for p in group['params'])
                self.log.f.info(f"参数组 {i} ({group['name']}): LR={group['lr']:.6f}, WD={group['weight_decay']:.6f}, 参数数={param_count}")
            if len(param_groups) >= 3:
                shared_target_lr = self.config.max_lr
                energy_target_lr = self.config.energy_lr
                force_target_lr = self.config.force_lr
                if force_target_lr > energy_target_lr > shared_target_lr:
                    self.log.f.info("✅ 学习率层级关系正确: force_lr > energy_lr > shared_lr")
                    self.log.f.info(f"  目标学习率: shared={shared_target_lr:.2e}, energy={energy_target_lr:.2e}, force={force_target_lr:.2e}")
                else:
                    self.log.f.warning("⚠️ 学习率层级关系异常，请检查配置")
                    self.log.f.info(f"  当前学习率: shared={param_groups[0]['lr']:.2e}, energy={param_groups[1]['lr']:.2e}, force={param_groups[2]['lr']:.2e}")
                    self.log.f.info(f"  目标学习率: shared={shared_target_lr:.2e}, energy={energy_target_lr:.2e}, force={force_target_lr:.2e}")
            if self.config.lr_scheduler == "differential_cosine":
                param_groups_lr = []
                for i, group in enumerate(param_groups):
                    if group['name'] == 'shared_layers':
                        param_groups_lr.append(self.config.max_lr)
                    elif group['name'] == 'energy_head':
                        param_groups_lr.append(self.config.energy_lr)
                    elif group['name'] == 'force_head':
                        param_groups_lr.append(self.config.force_lr)
                    elif group['name'] == 'output_other':
                        param_groups_lr.append(self.config.max_lr)
                    else:
                        param_groups_lr.append(self.config.max_lr)
                self.lr_scheduler = resolve_lr_scheduler(
                    sched_type=self.config.lr_scheduler,
                    optimizer=self.optimizer,
                    max_lr=self.config.max_lr,
                    min_lr=self.config.min_lr,
                    max_epochs=self.config.max_epochs,
                    steps_per_epoch=len(self.train_loader),
                    warmup_epochs=self.config.warmup_epochs,
                    param_groups_lr=param_groups_lr,
                    **self.config.lr_sche_kwargs,
                )
                self.is_differential_scheduler = True
                self.log.f.info(f"✅ 差分学习率调度器创建成功，参数组学习率: {param_groups_lr}")
                self.log.f.info("🎯 目标学习率映射:")
                for i, (group, target_lr) in enumerate(zip(param_groups, param_groups_lr)):
                    self.log.f.info(f"  参数组 {i} ({group['name']}): 目标LR={target_lr:.2e}")
                self.log.f.info(f"🔥 Warmup配置: {self.config.warmup_epochs} epochs")
                self.log.f.info(f"📈 学习率变化: min_lr={self.config.min_lr:.2e} → 目标学习率")
                self.log.f.info(f"📉 余弦退火: {self.config.max_epochs - self.config.warmup_epochs} epochs")
                self.log.f.info("🎯 预期学习率变化:")
                self.log.f.info(f"  Epoch 1-{self.config.warmup_epochs}: Warmup (线性增长)")
                self.log.f.info(f"  Epoch {self.config.warmup_epochs + 1}-{self.config.max_epochs}: 余弦退火 (平滑下降)")
                self.log.f.info("💡 Warmup策略说明:")
                self.log.f.info("  所有参数组从min_lr开始，避免训练初期不稳定")
                self.log.f.info("  Warmup期间线性增长到目标学习率")
                self.log.f.info("  之后保持差分学习率比例进行余弦退火")
                self.log.f.info("📊 学习率变化数值:")
                self.log.f.info(f"  Epoch 1: min_lr={self.config.min_lr:.2e}")
                self.log.f.info(f"  Epoch {self.config.warmup_epochs}: 目标学习率")
                self.log.f.info(f"  Epoch {self.config.max_epochs}: min_lr={self.config.min_lr:.2e}")
            else:
                self.lr_scheduler = resolve_lr_scheduler(
                    sched_type=self.config.lr_scheduler,
                    optimizer=self.optimizer,
                    max_lr=self.config.max_lr,
                    min_lr=self.config.min_lr,
                    max_epochs=self.config.max_epochs,
                    steps_per_epoch=len(self.train_loader),
                    warmup_epochs=self.config.warmup_epochs,
                    **self.config.lr_sche_kwargs,
                )
                self.is_differential_scheduler = False
        else:
            self.log.f.info("使用统一学习率模式")
            if self.config.optimizer == "adamW":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.max_lr,
                    **self.config.optim_kwargs
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            self.lr_scheduler = resolve_lr_scheduler(
                sched_type=self.config.lr_scheduler,
                optimizer=self.optimizer,
                max_lr=self.config.max_lr,
                min_lr=self.config.min_lr,
                max_epochs=self.config.max_epochs,
                steps_per_epoch=len(self.train_loader),
                warmup_epochs=self.config.warmup_epochs,
                **self.config.lr_sche_kwargs,
            )
            self.log.f.info(f"统一学习率: {self.config.max_lr}")
            self.log.f.info(f"权重衰减: {self.config.optim_kwargs.get('weight_decay', 'N/A')}")
def adaptive_ema_update(ema_model, current_model, decay):
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
