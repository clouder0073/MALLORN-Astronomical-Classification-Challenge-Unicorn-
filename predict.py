import pandas as pd
import numpy as np
import extinction
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import joblib
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import theilslopes
import warnings
warnings.filterwarnings('ignore')

print("=== TDE测试集预测 ===")

# 配置参数
BASE_PATH = './data/'
EFF_WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}
R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# 1. 加载训练好的模型
print("--- 加载模型 ---")
model_info = joblib.load('optimized_tde_model.pkl')
selected_features = model_info['features']
best_threshold = model_info['best_threshold']
models = model_info['models']

print(f"模型类型: {model_info['strategy']}")
print(f"特征数量: {len(selected_features)}")
print(f"最佳阈值: {best_threshold:.3f}")
print(f"模型数量: {len(models)}")

# 2. 加载测试集元数据
print("\n--- 加载测试集数据 ---")
test_log = pd.read_csv(f'{BASE_PATH}test_log.csv')
print(f"测试集对象数量: {len(test_log)}")

# 3. 测试集特征工程函数（与训练集保持一致）
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
    """提取改进的稳健特征（与训练集相同）"""
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

# 4. 处理测试集光变数据
def process_test_lightcurves():
    """处理测试集光变数据并提取特征"""
    
    print("\n--- 处理测试集光变数据 ---")
    
    # 检查测试集文件结构
    test_lc_paths = []
    
    # 检查是否存在单个测试集文件
    single_test_path = f'{BASE_PATH}test_lightcurves/test_full_lightcurves.csv'
    if os.path.exists(single_test_path):
        test_lc_paths = [single_test_path]
        print("找到单个测试集文件")
    else:
        # 检查分split的测试集文件
        for i in range(1, 21):
            split_path = f'{BASE_PATH}split_{i:02d}/test_full_lightcurves.csv'
            if os.path.exists(split_path):
                test_lc_paths.append(split_path)
        print(f"找到 {len(test_lc_paths)} 个测试集分片文件")
    
    if not test_lc_paths:
        raise FileNotFoundError("未找到测试集光变数据文件")
    
    all_test_features = []
    
    for lc_path in tqdm(test_lc_paths, desc="处理测试集分片"):
        # 加载光变数据
        lc = pd.read_csv(lc_path)
        lc.rename(columns={'Time (MJD)': 'mjd', 'Flux': 'flux', 'Flux_err': 'flux_err'}, inplace=True)
        
        # 只处理在test_log中存在的object_id
        relevant_objects = test_log['object_id'].unique()
        lc = lc[lc['object_id'].isin(relevant_objects)]
        
        if lc.empty:
            continue
        
        processed_dfs = []
        
        # 对每个object_id进行流量校正
        for object_id in lc['object_id'].unique():
            object_lc = lc[lc['object_id'] == object_id].copy()
            if object_lc.empty:
                continue
                
            # 获取该对象的EBV值
            object_ebv = test_log[test_log['object_id'] == object_id]['EBV'].iloc[0]
            A_v = R_V * object_ebv
            
            flux_corrected_list = []
            for _, row in object_lc.iterrows():
                A_lambda = extinction.fitzpatrick99(
                    np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V
                )[0]
                flux_corrected_list.append(row['flux'] * 10**(0.4 * A_lambda))
            
            object_lc['flux_corrected'] = flux_corrected_list
            processed_dfs.append(object_lc)
        
        if not processed_dfs:
            continue
            
        df = pd.concat(processed_dfs)
        
        # 计算距离和绝对星等（使用测试集的Z，注意测试集Z可能有误差）
        df['Z'] = df['object_id'].map(test_log.set_index('object_id')['Z'])
        z_values = df['Z'].fillna(0).values
        z_values[z_values <= 0] = 1e-6
        
        # 计算距离
        dist_pc = cosmo.luminosity_distance(z_values).to(u.pc).value
        df['distance_pc'] = dist_pc
        
        # 计算绝对星等
        df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
        df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
        df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)
        
        # 基础计算
        df['snr'] = df['flux_corrected'] / df['flux_err']
        df = df.sort_values(['object_id', 'mjd'])
        
        # 特征工程
        grouped = df.groupby('object_id')
        
        # 基础特征
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
        
        # 绝对星等特征
        abs_mag_agg = grouped.agg(
            abs_mag_min=('absolute_mag', 'min'),
            abs_mag_max=('absolute_mag', 'max'),
            abs_mag_mean=('absolute_mag', 'mean'),
            abs_mag_std=('absolute_mag', 'std'),
            abs_mag_span=('absolute_mag', lambda x: x.max() - x.min())
        )
        agg_features = agg_features.join(abs_mag_agg, how='left')
        
        # 时间加权斜率
        def weighted_slope(group):
            group = group.sort_values('mjd')
            delta_mjd = np.diff(group['mjd'])
            delta_flux = np.diff(group['flux_corrected'])
            slopes = delta_flux / (delta_mjd + 1e-9)
            return np.mean(slopes) if len(slopes) > 0 else 0
        
        agg_features['weighted_slope_mean'] = grouped.apply(weighted_slope)
        
        # 物理驱动特征
        agg_features['decay_alpha'] = grouped.apply(fit_decay_alpha)
        agg_features['is_single_peak'] = grouped.apply(is_single_peak)
        agg_features['rise_fall_ratio'] = grouped.apply(rise_fall_ratio)
        
        # 分波段特征
        pivot_mean = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='mean').add_suffix('_mean')
        pivot_std = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='std').add_suffix('_std')
        pivot_max = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='max').add_suffix('_max')
        agg_features = agg_features.join([pivot_mean, pivot_std, pivot_max], how='left')
        
        # 颜色特征
        for f in FILTERS:
            if f'{f}_mean' not in agg_features.columns: 
                agg_features[f'{f}_mean'] = np.nan
        agg_features['color_g_r_mean'] = agg_features['g_mean'] - agg_features['r_mean']
        agg_features['color_r_i_mean'] = agg_features['r_mean'] - agg_features['i_mean']
        agg_features['color_i_z_mean'] = agg_features['i_mean'] - agg_features['z_mean']
        
        # 新增改进特征
        improved_features = extract_improved_features(df)
        agg_features = agg_features.merge(improved_features, on='object_id', how='left')
        
        all_test_features.append(agg_features)
    
    if not all_test_features:
        raise ValueError("未生成任何测试集特征")
    
    # 合并所有特征
    test_feature_df = pd.concat(all_test_features)
    test_feature_df.reset_index(inplace=True)
    
    return test_feature_df

# 5. 生成测试集特征
test_features_df = process_test_lightcurves()

# 6. 合并元数据和特征
print("\n--- 合并测试集数据 ---")
test_final = test_log.merge(test_features_df, on='object_id', how='left')

# 检查缺失特征
missing_features = set(selected_features) - set(test_final.columns)
if missing_features:
    print(f"警告: 缺失 {len(missing_features)} 个特征，将用0填充")
    for feature in missing_features:
        test_final[feature] = 0

# 7. 准备预测数据
X_test = test_final[selected_features].copy()

# 处理缺失值（使用训练集的填充策略）
numeric_cols = X_test.select_dtypes(include=[np.number]).columns
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

std_cols = [c for c in X_test.columns if 'std' in c]
X_test[std_cols] = X_test[std_cols].fillna(0)

print(f"测试集特征形状: {X_test.shape}")

# 8. 进行预测
print("\n--- 进行预测 ---")
test_predictions = []

for i, model in enumerate(models):
    pred = model.predict_proba(X_test)[:, 1]
    test_predictions.append(pred)
    print(f"模型 {i+1} 预测完成")

# 平均预测
test_preds_mean = np.mean(test_predictions, axis=0)

# 应用最佳阈值
test_binary = (test_preds_mean >= best_threshold).astype(int)

print(f"测试集预测完成:")
print(f"平均预测概率: {test_preds_mean.mean():.4f} ± {test_preds_mean.std():.4f}")
print(f"正样本预测比例: {test_binary.mean():.4f}")
print(f"预测概率分布:")
for percentile in [10, 25, 50, 75, 90]:
    value = np.percentile(test_preds_mean, percentile)
    print(f"  {percentile}%分位数: {value:.4f}")

# 9. 生成提交文件
print("\n--- 生成提交文件 ---")
submission = pd.DataFrame({
    'object_id': test_final['object_id'],
    'predicted': test_binary
})

# 确保所有测试集对象都有预测
missing_objects = set(test_log['object_id']) - set(submission['object_id'])
if missing_objects:
    print(f"警告: {len(missing_objects)} 个对象没有预测，将设为0")
    missing_df = pd.DataFrame({
        'object_id': list(missing_objects),
        'predicted': 0
    })
    submission = pd.concat([submission, missing_df], ignore_index=True)

# 按object_id排序
submission = submission.sort_values('object_id')

# 保存提交文件
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ 提交文件已生成: '{submission_file}'")
print(f"提交文件形状: {submission.shape}")
print(f"正样本预测数量: {submission['predicted'].sum()}")
print(f"正样本比例: {submission['predicted'].mean():.4f}")

# 10. 保存详细预测结果
detailed_predictions = pd.DataFrame({
    'object_id': test_final['object_id'],
    'prediction_prob': test_preds_mean,
    'predicted': test_binary
})
detailed_predictions.to_csv('test_detailed_predictions.csv', index=False)
print("详细预测结果已保存: 'test_detailed_predictions.csv'")

# 11. 预测分析
print("\n--- 预测分析 ---")
print(f"使用的阈值: {best_threshold}")
print(f"预测概率范围: [{test_preds_mean.min():.4f}, {test_preds_mean.max():.4f}]")
print(f"高置信度正样本 (概率 > 0.7): {(test_preds_mean > 0.7).sum()}")
print(f"高置信度负样本 (概率 < 0.3): {(test_preds_mean < 0.3).sum()}")

# 检查特征重要性一致性
if hasattr(models[0], 'feature_importances_'):
    feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10最重要特征在测试集的表现:")
    print(importance_df.head(10))

print("\n🎉 测试集预测完成！")
print("📤 请提交 'submission.csv' 到Kaggle")
print(f"📊 预期LB分数应该接近: {model_info['oof_score']:.4f} (基于OOF F1)")
print("💡 如果LB分数不理想，请检查测试集与训练集的分布差异")