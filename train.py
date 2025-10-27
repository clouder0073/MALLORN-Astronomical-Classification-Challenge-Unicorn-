import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== TDE分类模型精细调优 ===")

# 1. 加载数据
print("--- 加载数据 ---")
df = pd.read_csv('processed_train_features_improved.csv')
features_to_exclude = ['object_id', 'SpecType', 'English Translation', 'split', 'target']
all_features = [col for col in df.columns if col not in features_to_exclude]
X = df[all_features]
y = df['target']
groups = df['split']

print(f"数据形状: {X.shape}")
print(f"正样本比例: {y.mean():.4f}")

# 2. 基于之前结果选择最佳特征子集
def select_optimal_features(df):
    """基于之前实验选择最佳特征"""
    
    # 从之前结果中表现好的特征
    high_importance_features = [
        'flux_abs_q25', 'flux_abs_q10', 'decay_alpha', 'r_max', 'mjd_span',
        'Z', 'flux_abs_q50', 'u_mean', 'i_max', 'u_max', 'peakiness',
        'color_r_i_mean', 'g_max', 'abs_mag_mean', 'abs_mag_min',
        'abs_mag_max', 'rise_fall_ratio', 'rise_fall_ratio_global',
        'asymmetry', 'u_std', 'positive_ratio', 'flux_abs_median',
        'variability_index', 'trend_stability', 'filter_coverage',
        'gap_max', 'obs_density', 'snr_mean', 'obs_count', 'flux_mean'
    ]
    
    # 只选择存在的特征
    selected = [f for f in high_importance_features if f in df.columns]
    print(f"选择了 {len(selected)} 个高重要性特征")
    return selected

print("\n--- 特征选择 ---")
selected_features = select_optimal_features(df)
X_optimal = df[selected_features]

# 3. 多策略模型训练
def train_multiple_strategies(X, y, groups):
    """尝试多种训练策略"""
    
    strategies = {}
    pos_weight = len(y) / (2 * np.sum(y))
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    # 策略1: 中等正则化
    print("\n--- 策略1: 中等正则化 ---")
    models1, oof1, scores1 = train_strategy(X, y, groups, {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'scale_pos_weight': pos_weight
    }, "中等正则化")
    
    strategies['medium_reg'] = {
        'models': models1, 
        'oof_preds': oof1, 
        'scores': scores1,
        'params': '中等正则化'
    }
    
    # 策略2: 轻度正则化 + 早停
    print("\n--- 策略2: 轻度正则化 + 严格早停 ---")
    models2, oof2, scores2 = train_strategy(X, y, groups, {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'scale_pos_weight': pos_weight
    }, "轻度正则化", early_stopping_rounds=50)
    
    strategies['light_reg'] = {
        'models': models2, 
        'oof_preds': oof2, 
        'scores': scores2,
        'params': '轻度正则化'
    }
    
    # 策略3: 集成学习 (LightGBM + RandomForest)
    print("\n--- 策略3: 模型集成 ---")
    ensemble_preds = np.zeros(len(X))
    ensemble_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"集成训练 Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42 + fold,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        
        # RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42 + fold,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict_proba(X_val)[:, 1]
        
        # 平均预测
        ensemble_pred = (lgb_preds + rf_preds) / 2
        ensemble_preds[val_idx] = ensemble_pred
        
        # 计算该fold的F1
        best_f1_fold = 0
        for thresh in np.arange(0.1, 0.9, 0.02):
            f1 = f1_score(y_val, (ensemble_pred >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
        
        ensemble_scores.append(best_f1_fold)
        print(f"Fold {fold + 1} F1: {best_f1_fold:.4f}")
    
    strategies['ensemble'] = {
        'models': None,  # 集成模型不保存单个模型
        'oof_preds': ensemble_preds, 
        'scores': ensemble_scores,
        'params': 'LightGBM+RF集成'
    }
    
    return strategies

def train_strategy(X, y, groups, params, strategy_name, early_stopping_rounds=100):
    """单个策略的训练"""
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    models = []
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"{strategy_name} - Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params, verbosity=-1, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # 寻找最佳阈值
        best_f1_fold = 0
        for thresh in np.arange(0.1, 0.9, 0.02):
            f1 = f1_score(y_val, (val_preds >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
        
        fold_scores.append(best_f1_fold)
        models.append(model)
        print(f"Fold {fold + 1} F1: {best_f1_fold:.4f}")
    
    return models, oof_preds, fold_scores

# 4. 评估所有策略
def evaluate_strategies(strategies, y):
    """评估所有训练策略"""
    
    results = []
    best_strategy = None
    best_f1 = 0
    
    for name, strategy in strategies.items():
        oof_preds = strategy['oof_preds']
        
        # 阈值优化
        best_threshold = 0.5
        best_f1_score = 0
        for thresh in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y, (oof_preds >= thresh).astype(int))
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = thresh
        
        # 计算其他指标
        binary_preds = (oof_preds >= best_threshold).astype(int)
        precision = np.mean(y[binary_preds == 1]) if np.sum(binary_preds) > 0 else 0
        recall = np.mean(binary_preds[y == 1])
        
        # 过拟合分析（仅对单个模型策略）
        if strategy['models'] is not None:
            train_f1 = analyze_overfitting(strategy['models'], X_optimal, y, best_threshold)
            overfitting_gap = train_f1 - best_f1_score
        else:
            train_f1 = np.nan
            overfitting_gap = np.nan
        
        results.append({
            'strategy': name,
            'params': strategy['params'],
            'oof_f1': best_f1_score,
            'precision': precision,
            'recall': recall,
            'threshold': best_threshold,
            'train_f1': train_f1,
            'overfitting_gap': overfitting_gap,
            'fold_scores': strategy['scores']
        })
        
        if best_f1_score > best_f1:
            best_f1 = best_f1_score
            best_strategy = name
    
    return pd.DataFrame(results), best_strategy

def analyze_overfitting(models, X, y, threshold):
    """分析过拟合"""
    train_preds = []
    for model in models:
        train_pred = model.predict_proba(X)[:, 1]
        train_preds.append(train_pred)
    
    train_preds_mean = np.mean(train_preds, axis=0)
    return f1_score(y, (train_preds_mean >= threshold).astype(int))

# 5. 主训练流程
print("\n--- 开始多策略训练 ---")
strategies = train_multiple_strategies(X_optimal, y, groups)

print("\n--- 策略评估 ---")
results_df, best_strategy_name = evaluate_strategies(strategies, y)

print("\n=== 所有策略结果 ===")
for _, row in results_df.iterrows():
    print(f"\n{row['strategy']} ({row['params']}):")
    print(f"  OOF F1: {row['oof_f1']:.4f}")
    print(f"  精确率: {row['precision']:.4f}")
    print(f"  召回率: {row['recall']:.4f}")
    print(f"  阈值: {row['threshold']:.3f}")
    if not pd.isna(row['overfitting_gap']):
        print(f"  过拟合差距: {row['overfitting_gap']:.4f}")
    print(f"  各Fold F1: {[f'{s:.4f}' for s in row['fold_scores']]}")

print(f"\n🎯 最佳策略: {best_strategy_name}")
best_strategy = strategies[best_strategy_name]

# 6. 最终模型优化
print("\n--- 最终模型优化 ---")

# 如果最佳策略是集成学习，我们需要重新训练一个可保存的版本
if best_strategy_name == 'ensemble':
    print("最佳策略是集成学习，训练可保存的版本...")
    
    # 训练最终的LightGBM模型，使用最佳参数
    final_models = []
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_optimal, y, groups)):
        print(f"最终训练 Fold {fold + 1}")
        
        X_train, X_val = X_optimal.iloc[train_idx], X_optimal.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate= 0.05 if best_strategy_name == 'medium_reg' else 0.1,
            num_leaves= 31 if best_strategy_name == 'medium_reg' else 63,
            max_depth= 7 if best_strategy_name == 'medium_reg' else 8,
            min_child_samples= 20 if best_strategy_name == 'medium_reg' else 10,
            subsample= 0.8 if best_strategy_name == 'medium_reg' else 0.9,
            colsample_bytree= 0.8 if best_strategy_name == 'medium_reg' else 0.9,
            reg_alpha= 0.3 if best_strategy_name == 'medium_reg' else 0.1,
            reg_lambda= 0.5 if best_strategy_name == 'medium_reg' else 0.2,
            scale_pos_weight= len(y) / (2 * np.sum(y)),
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        final_models.append(model)
    
    best_models = final_models
else:
    best_models = best_strategy['models']

# 7. 保存最佳模型
print("\n--- 保存最佳模型 ---")
best_model_info = {
    'models': best_models,
    'features': selected_features,
    'best_threshold': results_df[results_df['strategy'] == best_strategy_name]['threshold'].iloc[0],
    'oof_score': results_df[results_df['strategy'] == best_strategy_name]['oof_f1'].iloc[0],
    'strategy': best_strategy_name,
    'feature_names': selected_features
}

joblib.dump(best_model_info, 'optimized_tde_model.pkl')

# 保存详细结果
results_df.to_csv('training_strategies_comparison.csv', index=False)

# 保存OOF预测
best_oof_preds = best_strategy['oof_preds']
oof_results = pd.DataFrame({
    'object_id': df['object_id'],
    'true_target': y,
    'oof_prediction': best_oof_preds,
    'oof_binary': (best_oof_preds >= best_model_info['best_threshold']).astype(int)
})
oof_results.to_csv('optimized_oof_predictions.csv', index=False)

print("✅ 精细调优完成!")
print(f"📊 最佳策略: {best_strategy_name}")
print(f"🎯 最佳OOF F1: {best_model_info['oof_score']:.4f}")
print(f"⚖️  最佳阈值: {best_model_info['best_threshold']:.3f}")
print(f"📁 模型已保存: 'optimized_tde_model.pkl'")
print(f"📁 策略比较已保存: 'training_strategies_comparison.csv'")

# 8. 关键洞察
print("\n--- 关键洞察 ---")
print("当前问题分析:")
print("1. 严重过拟合 - 需要找到正则化和模型复杂度的平衡点")
print("2. 数据不平衡 - 正样本只有148个，需要更好的采样策略")
print("3. 特征质量 - 需要确保特征具有判别力")

print("\n建议下一步:")
if best_model_info['oof_score'] < 0.38:
    print("⚠️  OOF F1仍然较低，建议:")
    print("   - 重新检查特征工程")
    print("   - 尝试过采样技术(SMOTE)")
    print("   - 考虑使用神经网络")
else:
    print("✅ 结果可接受，可以进行测试集预测")