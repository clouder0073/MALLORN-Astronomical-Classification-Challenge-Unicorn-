# MALLORN-Astronomical-Classification-Challenge-Unicorn
This is the approach used by my team (Unicorn) in the Kaggle community competition — the MALLORN Astronomical Classification Challenge, which includes three steps: data preprocessing and feature engineering, training and hyperparameter tuning, and prediction.

1 PreprocessandFeatures.py
2 train.py
3 predict.py


# Method Description

Feature Engineering:

We designed and implemented a physics-informed feature engineering pipeline that incorporates redshift correction, luminosity distance calculation, and interstellar extinction correction. This pipeline introduces absolute magnitudes, attenuation index fitting, multi-band color features, and robust time-series morphological features. Key discriminative features were retained based on information-gain-driven feature importance analysis.

Model Training and Hyperparameter Tuning:

We built a LightGBM-based time-series classification model and employed Optuna for automated hyperparameter optimization, systematically tuning both model parameters and decision thresholds with the F1 score as the objective function. To address the severe class imbalance (positive samples account for only 4.8%), we adopted a grouped cross-validation strategy to ensure model generalization. Additionally, multi-level regularization techniques were introduced to mitigate overfitting.
