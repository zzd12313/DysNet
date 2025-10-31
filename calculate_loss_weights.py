
import h5py
import numpy as np
import json
import os
from pathlib import Path
import sys
sys.path.append('/root/XequiNet')
def calculate_training_stats():
    print("=== 计算训练集归一化统计值 ===")
    dataset_path = "/root/autodl-tmp/raw/SPICE_PUBCHEM_SUBSETS_SPLITS_CLEANED.hdf5"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return None
    all_energies = []
    all_forces = []
    print("正在分析训练集数据...")
    try:
        with h5py.File(dataset_path, 'r') as f:
            print(f"数据集结构: {list(f.keys())}")
            print(f"splits结构: {list(f['splits'].keys())}")
            train_molecules = list(f['splits']['train'].keys())
            print(f"训练集分子数: {len(train_molecules)}")
            for i, mol_key in enumerate(train_molecules):
                if i % 1000 == 0:
                    print(f"处理分子 {i}/{len(train_molecules)}")
                mol_data = f['splits']['train'][mol_key]
                try:
                    mol_energies = mol_data['dft_total_energy'][:]
                    all_energies.extend(mol_energies)
                    mol_forces = mol_data['dft_total_gradient'][:]
                    all_forces.extend(mol_forces.flatten())
                except Exception as e:
                    print(f"分子 {mol_key} 处理失败: {e}")
                    continue
    except Exception as e:
        print(f"❌ 读取数据集失败: {e}")
        return None
    all_energies = np.array(all_energies)
    all_forces = np.array(all_forces)
    print(f"\n=== 训练集统计结果 ===")
    print(f"能量样本数: {len(all_energies)}")
    print(f"力样本数: {len(all_forces)}")
    energy_mean = np.mean(all_energies)
    energy_std = np.std(all_energies)
    force_mean = np.mean(all_forces)
    force_std = np.std(all_forces)
    print(f"\n能量统计:")
    print(f"  均值 (target_mean): {energy_mean:.6f}")
    print(f"  标准差 (target_std): {energy_std:.6f}")
    print(f"  最小值: {np.min(all_energies):.6f}")
    print(f"  最大值: {np.max(all_energies):.6f}")
    print(f"  绝对均值: {np.mean(np.abs(all_energies)):.6f}")
    print(f"\n力统计:")
    print(f"  均值 (grad_target_mean): {force_mean:.6f}")
    print(f"  标准差 (grad_target_std): {force_std:.6f}")
    print(f"  最小值: {np.min(all_forces):.6f}")
    print(f"  最大值: {np.max(all_forces):.6f}")
    print(f"  绝对均值: {np.mean(np.abs(all_forces)):.6f}")
    print(f"\n=== 物理合理性检查 ===")
    if abs(force_mean) < 1e-6:
        print(f"✅ 力均值接近0 ({force_mean:.2e})，符合物理预期")
    else:
        print(f"⚠️  力均值不为0 ({force_mean:.6f})，可能需要检查数据")
    if energy_std > 0 and force_std > 0:
        print(f"✅ 能量和力标准差都为正，可以安全归一化")
    else:
        print(f"❌ 标准差为0，无法归一化")
        return None
    return {
        'energy_mean': energy_mean,
        'energy_std': energy_std,
        'force_mean': force_mean,
        'force_std': force_std
    }
def calculate_loss_weights(stats):
    print(f"\n=== 计算损失权重 ===")
    energy_std = stats['energy_std']
    force_std = stats['force_std']
    print(f"原始能量标准差: {energy_std:.6f}")
    print(f"原始力标准差: {force_std:.6f}")
    weight_ratio = energy_std / force_std
    print(f"基于标准差的权重比例: {weight_ratio:.6f}")
    energy_abs_mean = abs(stats['energy_mean'])
    force_abs_mean = abs(stats['force_mean'])
    if force_abs_mean > 0:
        weight_ratio_abs = energy_abs_mean / force_abs_mean
        print(f"基于绝对均值的权重比例: {weight_ratio_abs:.6f}")
    else:
        weight_ratio_abs = weight_ratio
    energy_coefficient = 1.0
    force_coefficient = weight_ratio
    print(f"\n=== 推荐权重设置 ===")
    print(f"energy_coefficient: {energy_coefficient}")
    print(f"force_coefficient: {force_coefficient:.6f}")
    if force_coefficient > 100:
        print(f"⚠️  力权重 {force_coefficient:.6f} 可能过高，建议限制在100以内")
        force_coefficient = 100.0
    elif force_coefficient < 0.01:
        print(f"⚠️  力权重 {force_coefficient:.6f} 可能过低，建议至少为0.01")
        force_coefficient = 0.01
    print(f"最终推荐的力权重: {force_coefficient:.6f}")
    return energy_coefficient, force_coefficient
def update_config(stats, energy_coefficient, force_coefficient):
    print(f"\n=== 更新配置文件 ===")
    config_path = 'config_spice_energy_force.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['normalize_labels'] = True
    config['target_mean'] = float(stats['energy_mean'])
    config['target_std'] = float(stats['energy_std'])
    config['grad_target_mean'] = float(stats['force_mean'])
    config['grad_target_std'] = float(stats['force_std'])
    config['energy_coefficient'] = energy_coefficient
    config['force_coefficient'] = force_coefficient
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ 配置文件已更新:")
    print(f"  归一化设置:")
    print(f"    normalize_labels: {config['normalize_labels']}")
    print(f"    target_mean: {config['target_mean']:.6f}")
    print(f"    target_std: {config['target_std']:.6f}")
    print(f"    grad_target_mean: {config['grad_target_mean']:.6f}")
    print(f"    grad_target_std: {config['grad_target_std']:.6f}")
    print(f"  损失权重:")
    print(f"    energy_coefficient: {config['energy_coefficient']}")
    print(f"    force_coefficient: {config['force_coefficient']:.6f}")
def verify_normalization():
    print(f"\n=== 验证归一化效果 ===")
    with open('config_spice_energy_force.json', 'r') as f:
        config = json.load(f)
    print(f"归一化参数:")
    print(f"  能量: (x - {config['target_mean']:.6f}) / {config['target_std']:.6f}")
    print(f"  力: (x - {config['grad_target_mean']:.6f}) / {config['grad_target_std']:.6f}")
    print(f"\n归一化示例:")
    energy_examples = [config['target_mean'], config['target_mean'] + config['target_std'], config['target_mean'] - config['target_std']]
    for energy in energy_examples:
        normalized = (energy - config['target_mean']) / config['target_std']
        print(f"  能量 {energy:.6f} -> {normalized:.6f}")
    force_examples = [config['grad_target_mean'], config['grad_target_mean'] + config['grad_target_std'], config['grad_target_mean'] - config['grad_target_std']]
    for force in force_examples:
        normalized = (force - config['grad_target_mean']) / config['grad_target_std']
        print(f"  力 {force:.6f} -> {normalized:.6f}")
def main():
    print("🚀 开始基于Gemini方法的归一化预处理...")
    stats = calculate_training_stats()
    if stats is None:
        print("❌ 统计值计算失败")
        return
    energy_coef, force_coef = calculate_loss_weights(stats)
    update_config(stats, energy_coef, force_coef)
    verify_normalization()
    print(f"\n🎉 归一化预处理完成！")
    print(f"现在可以开始训练，数据将自动进行正确的归一化处理。")
if __name__ == "__main__":
    main()
