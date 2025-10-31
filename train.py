import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig, ZeroLogger, set_default_unit,
    calculate_stats, distributed_zero_first,
)
from xequinet.utils.functional import calculate_force_stats
from xequinet.data import create_dataset
def main():
    parser = argparse.ArgumentParser(description="XequiNet distributed training script")
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    parser.add_argument(
        "--resume", "-r", type=str, default=None,
        help="Resume training from checkpoint file",
    )
    parser.add_argument(
        "--resume_ema", type=str, default=None,
        help="Resume training from EMA checkpoint file",
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    args = parser.parse_args()
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")
    if os.path.isfile(args.config):
        with open(args.config, "r") as json_file:
            config = NetConfig.model_validate_json(json_file.read())
    else:
        Warning(f"Config file {args.config} not found. Default config will be used.")
        config = NetConfig()
    if args.resume:
        config.ckpt_file = args.resume
        config.resume = True
    elif args.resume_ema:
        config.ckpt_file = args.resume_ema
        config.resume = True
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    is_rank0 = (local_rank == 0)
    log = ZeroLogger(is_rank0=is_rank0, output_dir=config.save_dir, log_file=config.log_file)
    if config.output_mode == "conservative" and config.ckpt_file:
        log.s.info("=== 第二阶段：保守力微调训练 ===")
        log.s.info("策略：从直接力预测切换到保守力预测")
        log.s.info(f"预训练检查点：{config.ckpt_file}")
        log.s.info("架构：单通道（只预测能量，力通过自动微分）")
        log.s.info("训练方式：权重训练 energy_weight = 0.9, force_weight = 0.1")
        log.s.info("学习率策略：统一学习率，不使用差分学习率")
        log.s.info("物理原理：传统保守力训练方法，能量为主，力为辅")
        log.s.info("力通过自动微分计算：F = -∇E")
    if config.version in ["xpainn-multibody-v2", "dysnet"]:
        log.s.info("使用 DysNet 配置:")
        log.s.info(f"  - 等变门控: {config.use_equivariant_gate}")
        log.s.info(f"  - 球谐消息: {config.use_spherical_message}")
        log.s.info(f"  - 不变性归一化: {config.use_invariant_norm}")
        log.s.info(f"  - 度数平衡: {getattr(config, 'std_balance_degrees', True)}")
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
    if config.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif config.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise NotImplementedError(f"Unknown default data type: {config.default_dtype}")
    set_default_unit(config.default_property_unit, config.default_length_unit)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(device)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if config.dataset_type == "spice" and local_rank == 0:
        log.s.info("=== SPICE数据集预处理 ===")
        log.s.info("检查预处理文件是否存在...")
        processed_dir = "/root/autodl-tmp/processed"
        base_name = os.path.splitext(config.data_files)[0]
        train_file = os.path.join(processed_dir, f"{base_name}_train.pt")
        val_file = os.path.join(processed_dir, f"{base_name}_val.pt")
        test_file = os.path.join(processed_dir, f"{base_name}_test.pt")
        files_exist = all(os.path.exists(f) for f in [train_file, val_file, test_file])
        if not files_exist:
            log.s.info("预处理文件不存在，开始预处理...")
            from xequinet.data.spice_dataset import SPICEDataset
            splits = ["train", "val", "test"]
            from tqdm import tqdm
            log.s.info("开始预处理所有数据集划分...")
            for split in tqdm(splits, desc="预处理数据集", unit="划分"):
                log.s.info(f"处理 {split} 划分...")
                try:
                    dataset = SPICEDataset(
                        h5_file_path=os.path.join(config.data_root, config.data_files),
                        split=split,
                        cutoff=config.cutoff,
                        target_mean=config.target_mean,
                        target_std=config.target_std,
                        grad_target_mean=config.grad_target_mean,
                        grad_target_std=config.grad_target_std,
                        coord_unit=config.default_length_unit,
                        cutoff_unit=config.default_length_unit,
                    )
                    actual_samples = len(dataset.data_list) if hasattr(dataset, 'data_list') else len(dataset)
                    log.s.info(f"{split} 划分预处理完成，样本数: {actual_samples}")
                    os.makedirs(processed_dir, exist_ok=True)
                    base_name = os.path.splitext(config.data_files)[0]
                    save_file = os.path.join(processed_dir, f"{base_name}_{split}.pt")
                    torch.save(dataset.data_list, save_file)
                    log.s.info(f"预处理文件已保存到: {save_file}")
                except Exception as e:
                    log.s.error(f"{split} 划分预处理失败: {e}")
                    raise
            log.s.info("✅ 所有划分预处理完成！")
        else:
            log.s.info("预处理文件已存在，跳过预处理步骤")
    if config.dataset_type == "spice":
        dist.barrier()
    train_dataset = create_dataset(config, "train", local_rank)
    valid_dataset = create_dataset(config, "val", local_rank)
    train_sampler = DistributedSampler(train_dataset, world_size, local_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, world_size, local_rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size // world_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.vbatch_size // world_size, sampler=valid_sampler,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    with distributed_zero_first(local_rank):
        energy_mean, energy_std = calculate_stats(train_loader)
        prop_unit = getattr(config, 'default_property_unit', 'hartree')
        log.s.info(f"Energy Mean: {energy_mean:6.4f} {prop_unit}")
        log.s.info(f"Energy Std : {energy_std:6.4f} {prop_unit}")
        force_mean, force_std = calculate_force_stats(train_loader)
        force_unit = getattr(config, 'force_unit', 'hartree/bohr')
        log.s.info(f"Force Mean: {force_mean:6.4f} {force_unit}")
        log.s.info(f"Force Std : {force_std:6.4f} {force_unit}")
    model = resolve_model(config)
    log.s.info(model)
    model.to(device)
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=device,
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=False
    )
    n_params = 0
    for name, param in ddp_model.named_parameters():
        n_params += param.numel()
        log.s.info(f"{name}: {param.numel()}")
    log.s.info(f"Total number of parameters to be optimized: {n_params}")
    if config.output_mode == "grad" or config.output_mode == "conservative":
        from xequinet.utils import GradTrainer as MyTrainer
    else:
        from xequinet.utils import Trainer as MyTrainer
    trainer = MyTrainer(ddp_model, config, device, train_loader, valid_loader, train_sampler, log)
    trainer.start()
if __name__ == "__main__":
    main()
