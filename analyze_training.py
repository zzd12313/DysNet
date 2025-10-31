
import re
import numpy as np
def analyze_log_file(log_file_path):
    print("=== 分析训练日志 ===")
    energy_maes = []
    force_maes = []
    with open(log_file_path, 'r') as f:
        for line in f:
            train_match = re.search(r'Train MAE: Energy\s+([\d.]+)\s+hartree\s+Force\s+([\d.]+)\s+hartree', line)
            if train_match:
                energy_mae = float(train_match.group(1))
                force_mae = float(train_match.group(2))
                if not np.isnan(energy_mae) and not np.isnan(force_mae):
                    energy_maes.append(energy_mae)
                    force_maes.append(force_mae)
    if not energy_maes:
        print("未找到有效的训练数据")
        return
    print(f"找到 {len(energy_maes)} 个有效训练步骤")
    energy_mean = np.mean(energy_maes)
    energy_std = np.std(energy_maes)
    force_mean = np.mean(force_maes)
    force_std = np.std(force_maes)
    print(f"\n=== 训练损失统计 ===")
    print(f"能量MAE:")
    print(f"  均值: {energy_mean:.6f} hartree")
    print(f"  标准差: {energy_std:.6f} hartree")
    print(f"  范围: {min(energy_maes):.6f} - {max(energy_maes):.6f} hartree")
    print(f"\n力MAE:")
    print(f"  均值: {force_mean:.6f} hartree")
    print(f"  标准差: {force_std:.6f} hartree")
    print(f"  范围: {min(force_maes):.6f} - {max(force_maes):.6f} hartree")
    loss_ratio = energy_mean / force_mean
    print(f"\n=== 损失比例分析 ===")
    print(f"能量/力损失比例: {loss_ratio:.2f}")
    print(f"\n=== 推荐权重设置 ===")
    force_coef_1 = loss_ratio
    print(f"方法1 (直接比例): force_coefficient = {force_coef_1:.2f}")
    std_ratio = energy_std / force_std
    force_coef_2 = std_ratio
    print(f"方法2 (标准差比例): force_coefficient = {force_coef_2:.2f}")
    force_coef_3 = np.sqrt(loss_ratio * std_ratio)
    print(f"方法3 (几何平均): force_coefficient = {force_coef_3:.2f}")
    force_coef_4 = 50.0
    print(f"方法4 (保守估计): force_coefficient = {force_coef_4:.2f}")
    recommended_force_coef = min(force_coef_4, 100.0)
    print(f"\n推荐最终设置:")
    print(f"  energy_coefficient: 1.0")
    print(f"  force_coefficient: {recommended_force_coef:.2f}")
    return recommended_force_coef
def update_config_with_log_analysis(force_coefficient):
    print(f"\n=== 更新配置文件 ===")
    config_path = 'config_spice_energy_force.json'
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['energy_coefficient'] = 1.0
    config['force_coefficient'] = force_coefficient
    config['max_lr'] = 0.0002
    config['grad_clip'] = 0.05
    config['batch_size'] = 128
    config['warmup_epochs'] = 30
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置文件已更新:")
    print(f"  energy_coefficient: 1.0")
    print(f"  force_coefficient: {force_coefficient:.2f}")
    print(f"  max_lr: 0.0002")
    print(f"  grad_clip: 0.05")
    print(f"  batch_size: 128")
    print(f"  warmup_epochs: 30")
def main():
    log_file = "spice_energy_force.log"
    try:
        force_coef = analyze_log_file(log_file)
        if force_coef:
            update_config_with_log_analysis(force_coef)
            print(f"\n✅ 基于训练日志的配置更新完成！")
        else:
            print(f"\n❌ 无法从日志中提取有效数据")
    except Exception as e:
        print(f"分析失败: {e}")
if __name__ == "__main__":
    main()
