import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import extinction
from scipy.optimize import curve_fit
from scipy.stats import theilslopes
import os
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

# 配置
BASE_PATH = './data/'
NUM_SPLITS = 20
EFF_WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}
R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

print("正在加载主元数据文件 train_log.csv...")
train_log_full = pd.read_csv(f'{BASE_PATH}train_log.csv')

# ---------------------------------------------------------------------------------
# 预先计算所有天体的距离
# ---------------------------------------------------------------------------------
print("\n--- 开始计算天体物理特征 (距离与绝对星等) ---")
log_meta = train_log_full[['object_id', 'Z']].copy()

# 处理红移为0或负数的情况
z = log_meta['Z'].values
z[z <= 0] = 1e-6

print("正在计算光度距离...")
dist_pc = cosmo.luminosity_distance(z).to(u.pc).value
dist_map = dict(zip(log_meta['object_id'], dist_pc))
print("距离计算完成！")

# ---------------------------------------------------------------------------------
# 特征提取函数
# ---------------------------------------------------------------------------------
def tde_decay_model(t, alpha, t0, A):
    return A * (t - t0 + 1e-6) ** (-alpha)

def fit_decay_alpha(group):
    group = group.sort_values('mjd')
    t = group['mjd'].values
    f = group['flux_corrected'].values
    if len(f) < 3: return np.nan
    peak_idx = np.argmax(f)
    if peak_idx == len(f) - 1: return np.nan
    t_decay = t[peak_idx:] - t[peak_idx]
    f_decay = f[peak_idx:]
    try:
        popt, _ = curve_fit(
            tde_decay_model, t_decay, f_decay,
            p0=[1.5, 0, f[peak_idx]],
            bounds=([0.5, -10, 0], [3.0, 10, np.inf]),
            maxfev=500
        )
        alpha = popt[0]
        return alpha if 0.5 <= alpha <= 3.0 else np.nan
    except:
        return np.nan

def is_single_peak(group):
    f = group['flux_corrected'].values
    if len(f) < 3: return 0
    peaks = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
    return int(np.sum(peaks) == 1)

def rise_fall_ratio(group):
    group = group.sort_values('mjd')
    t = group['mjd'].values
    f = group['flux_corrected'].values
    if len(f) < 3: return np.nan
    peak_idx = np.argmax(f)
    if peak_idx == 0 or peak_idx == len(f) - 1: return np.nan
    rise_time = t[peak_idx] - t[0]
    fall_time = t[-1] - t[peak_idx]
    return rise_time / (fall_time + 1e-6)

def extract_improved_features(df):
    """提取改进的稳健特征"""
    features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id].sort_values('mjd')
        
        features = {}
        flux = obj_data['flux_corrected'].values
        mjd = obj_data['mjd'].values
        
        # 1. 负流量处理
        flux_positive = flux.clip(min=1e-9)
        flux_abs = np.abs(flux)
        
        # 分位数特征
        for q in [10, 25, 50, 75, 90]:
            features[f'flux_q{q}'] = np.percentile(flux_positive, q)
            features[f'flux_abs_q{q}'] = np.percentile(flux_abs, q)
        
        features['positive_ratio'] = (flux > 0).mean()
        features['flux_abs_median'] = np.median(flux_abs)
        
        # 2. 观测gap特征
        if len(mjd) > 1:
            mjd_diff = np.diff(mjd)
            features['gap_mean'] = np.mean(mjd_diff)
            features['gap_std'] = np.std(mjd_diff)
            features['gap_max'] = np.max(mjd_diff)
            features['gap_median'] = np.median(mjd_diff)
            features['obs_density'] = len(obj_data) / (mjd[-1] - mjd[0] + 1e-9)
        
        # 3. 波段特征
        unique_filters = obj_data['Filter'].nunique()
        features['filter_coverage'] = unique_filters / 6.0
        
        # 各波段的观测次数比例
        for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            band_count = (obj_data['Filter'] == band).sum()
            features[f'{band}_obs_ratio'] = band_count / len(obj_data)
        
        # 4. 信噪比特征
        snr = flux_positive / (obj_data['flux_err'].values + 1e-9)
        features['snr_median'] = np.median(snr)
        features['snr_q10'] = np.percentile(snr, 10)
        features['snr_q90'] = np.percentile(snr, 90)
        
        # 5. 改进的趋势稳定性（使用Theil-Sen）
        if len(flux) >= 5:
            try:
                x = np.arange(len(flux))
                slope, _, _, _ = theilslopes(flux, x)
                features['trend_slope'] = slope
                # 趋势稳定性：残差的相对大小
                predicted = slope * x + np.median(flux)
                residuals = flux - predicted
                features['trend_stability'] = 1.0 / (np.std(residuals) / (np.std(flux) + 1e-9) + 1e-9)
            except:
                features['trend_slope'] = 0
                features['trend_stability'] = 0
        else:
            features['trend_slope'] = 0
            features['trend_stability'] = 0
        
        # 6. 通用时序形态特征（不依赖标签）
        if len(flux) >= 3:
            # 变异性指数
            features['variability_index'] = np.std(flux) / (np.mean(flux_positive) + 1e-9)
            
            # 峰值性
            features['peakiness'] = np.max(flux_positive) / (np.median(flux_positive) + 1e-9)
            
            # 不对称性
            peak_idx = np.argmax(flux)
            if 0 < peak_idx < len(flux) - 1:
                before_peak = flux[:peak_idx]
                after_peak = flux[peak_idx+1:]
                std_before = np.std(before_peak) if len(before_peak) > 1 else 0
                std_after = np.std(after_peak) if len(after_peak) > 1 else 0
                features['asymmetry'] = std_after / (std_before + 1e-9)
            else:
                features['asymmetry'] = 0
            
            # 上升下降比率
            if len(flux) > 5:
                first_half = flux[:len(flux)//2]
                second_half = flux[len(flux)//2:]
                features['rise_fall_ratio_global'] = np.mean(first_half) / (np.mean(second_half) + 1e-9)
        
        features['object_id'] = object_id
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# ---------------------------------------------------------------------------------
# 主处理流程
# ---------------------------------------------------------------------------------
all_features_list = []

print(f"\n--- 开始处理 {NUM_SPLITS} 个训练数据分片 ---")
start_time = time.time()

for i in tqdm(range(1, NUM_SPLITS + 1), desc="总体进度"):
    split_id_str = str(i).zfill(2)
    
    lc = pd.read_csv(f'{BASE_PATH}split_{split_id_str}/train_full_lightcurves.csv')
    lc.rename(columns={'Time (MJD)': 'mjd', 'Flux': 'flux', 'Flux_err': 'flux_err'}, inplace=True)

    log = train_log_full[train_log_full['split'] == f'split_{split_id_str}']
    if log.empty or lc.empty: continue

    # 1. 流量校正
    processed_dfs = []
    for object_id in log['object_id'].unique():
        object_lc = lc[lc['object_id'] == object_id].copy()
        if object_lc.empty: continue
        object_log = log[log['object_id'] == object_id].iloc[0]
        ebv = object_log['EBV']
        A_v = R_V * ebv
        flux_corrected_list = []
        for index, row in object_lc.iterrows():
            A_lambda = extinction.fitzpatrick99(np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V)[0]
            flux_corrected_list.append(row['flux'] * 10**(0.4 * A_lambda))
        object_lc['flux_corrected'] = flux_corrected_list
        processed_dfs.append(object_lc)
    if not processed_dfs: continue
    df = pd.concat(processed_dfs)

    # 2. 距离和绝对星等
    df['distance_pc'] = df['object_id'].map(dist_map)
    df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
    df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
    df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)

    # 3. 基础计算 & 排序
    df['snr'] = df['flux_corrected'] / df['flux_err']
    df = df.sort_values(['object_id', 'mjd'])
    
    # 4. 特征工程
    grouped = df.groupby('object_id')
    
    # a. 全局特征
    agg_features = grouped.agg(
        flux_mean=('flux_corrected', 'mean'),
        flux_std=('flux_corrected', 'std'),
        flux_max=('flux_corrected', 'max'),
        flux_min=('flux_corrected', 'min'),
        flux_skew=('flux_corrected', 'skew'),
        mjd_span=('mjd', lambda x: x.max() - x.min()),
        snr_mean=('snr', 'mean'),
        snr_max=('snr', 'max'),
        snr_std=('snr', 'std'),
        obs_count=('mjd', 'size')
    )

    # 基于绝对星等的特征
    abs_mag_agg = grouped.agg(
        abs_mag_min=('absolute_mag', 'min'),
        abs_mag_max=('absolute_mag', 'max'),
        abs_mag_mean=('absolute_mag', 'mean'),
        abs_mag_std=('absolute_mag', 'std'),
        abs_mag_span=('absolute_mag', lambda x: x.max() - x.min())
    )
    agg_features = agg_features.join(abs_mag_agg, how='left')

    # b. 时间加权的斜率特征
    def weighted_slope(group):
        group = group.sort_values('mjd')
        delta_mjd = np.diff(group['mjd'])
        delta_flux = np.diff(group['flux_corrected'])
        slopes = delta_flux / (delta_mjd + 1e-9)
        return np.mean(slopes) if len(slopes) > 0 else 0
    agg_features['weighted_slope_mean'] = grouped.apply(weighted_slope)

    # c. 物理驱动特征
    agg_features['decay_alpha'] = grouped.apply(fit_decay_alpha)
    agg_features['is_single_peak'] = grouped.apply(is_single_peak)
    agg_features['rise_fall_ratio'] = grouped.apply(rise_fall_ratio)

    # d. 分波段特征
    pivot_mean = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='mean').add_suffix('_mean')
    pivot_std = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='std').add_suffix('_std')
    pivot_max = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='max').add_suffix('_max')
    agg_features = agg_features.join([pivot_mean, pivot_std, pivot_max], how='left')

    # e. 颜色特征
    for f in FILTERS:
        if f'{f}_mean' not in agg_features.columns: 
            agg_features[f'{f}_mean'] = np.nan
    agg_features['color_g_r_mean'] = agg_features['g_mean'] - agg_features['r_mean']
    agg_features['color_r_i_mean'] = agg_features['r_mean'] - agg_features['i_mean']
    agg_features['color_i_z_mean'] = agg_features['i_mean'] - agg_features['z_mean']

    # f. 新增改进特征
    improved_features = extract_improved_features(df)
    agg_features = agg_features.merge(improved_features, on='object_id', how='left')

    all_features_list.append(agg_features)

# 5. 最终整合
print("\n--- 所有分片处理完毕，开始最后整合 ---")
full_feature_df = pd.concat(all_features_list)
full_feature_df.reset_index(inplace=True)

# 6. 与元数据合并
final_df = pd.merge(train_log_full, full_feature_df, on='object_id', how='right')

# 7. 缺失值填充
std_cols = [c for c in final_df.columns if 'std' in c]
final_df[std_cols] = final_df[std_cols].fillna(0)
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
final_df[numeric_cols] = final_df[numeric_cols].fillna(final_df[numeric_cols].median())

# 8. 保存
OUTPUT_FILE = 'processed_train_features_improved.csv'
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- ✅ 数据处理全部完成！ ---")
print(f"总耗时: {(time.time() - start_time)/60:.2f} 分钟")
print(f"最终生成的训练特征数据形状: {final_df.shape}")
print(f"文件已保存到: '{OUTPUT_FILE}'")