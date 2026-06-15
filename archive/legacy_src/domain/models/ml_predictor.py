"""Machine learning predictor for JP ETF returns using LightGBM.

Defines feature construction, target variable definition (raw vs. Z-score),
and rolling training and prediction.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Any, Tuple

from data.ticker_registry import US_TICKERS, JP_TICKERS, TOPIX_TICKER

logger = logging.getLogger(__name__)


def compute_rolling_z_scores_us(
    df_exec: pd.DataFrame,
    us_tickers: List[str],
    corr_window: int = 60,
    ewma_half_life: float = 45.0,
) -> pd.DataFrame:
    """Compute rolling standardised US returns matching the strategy's EWMA logic.

    For each date, standardized return is: (r_t - mu_t) / sigma_t
    where mu_t and sigma_t are the EWMA mean and standard deviation computed over
    the `corr_window` prior days.
    """
    n_u = len(us_tickers)
    us_cols = [f"us_cc_{tk}" for tk in us_tickers]
    us_returns = df_exec[us_cols].values
    T = len(df_exec)
    z_us = np.zeros((T, n_u))

    decay = np.power(0.5, 1.0 / float(ewma_half_life))
    weights = np.power(decay, np.arange(corr_window - 1, -1, -1))
    weights = weights / np.sum(weights)

    for i in range(T):
        if i < corr_window:
            z_us[i] = np.nan
            continue
        # Window of historical returns prior to row i
        window_returns = us_returns[i - corr_window : i]
        mu = np.sum(window_returns * weights[:, None], axis=0)
        var = np.sum(((window_returns - mu) ** 2) * weights[:, None], axis=0)
        sigma = np.sqrt(np.maximum(var, 1e-16))
        sigma[sigma == 0] = 1e-8

        # Return at row i (sig_date t)
        r_us_t = us_returns[i]
        z_us[i] = (r_us_t - mu) / sigma

    z_us_df = pd.DataFrame(
        z_us, index=df_exec.index, columns=[f"z_us_{tk}" for tk in us_tickers]
    )
    return z_us_df


def compute_jp_volatility(
    df_exec: pd.DataFrame,
    jp_tickers: List[str],
    vol_window: int = 20,
) -> pd.DataFrame:
    """Compute the 20-day rolling standard deviation of Close-to-Close returns of JP ETFs.

    Note: `jp_cc_{tk}` in row i is the Close-to-Close return ending on sig_date t (which is known at t).
    So the 20-day standard deviation ending at row i (inclusive of row i) is indeed the volatility
    known at sig_date t, which we denote as sigma_{j, t}.
    """
    vol_dict = {}
    for tk in jp_tickers:
        col = f"jp_cc_{tk}"
        # ddof=1 to match standard sample standard deviation
        vol_dict[f"vol_20_{tk}"] = df_exec[col].rolling(vol_window).std(ddof=1)
    return pd.DataFrame(vol_dict, index=df_exec.index)


class LGBMPredictor:
    """LightGBM regression model wrapper with conservative hyperparameters."""

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 50,
        min_child_samples: int = 10,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_samples=min_child_samples,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LGBMPredictor:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MLRollingRunner:
    """Orchestrates rolling training and prediction for the 17 JP ETFs."""

    def __init__(
        self,
        df_exec: pd.DataFrame,
        us_tickers: List[str],
        jp_tickers: List[str],
        topix_ticker: str = "1306.T",
        train_window: int = 250,
        refit_interval: int = 1,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 50,
        min_child_samples: int = 10,
        random_seed: int = 42,
    ):
        self.df_exec = df_exec.copy()
        self.us_tickers = us_tickers
        self.jp_tickers = jp_tickers
        self.topix_ticker = topix_ticker
        self.train_window = train_window
        self.refit_interval = refit_interval
        self.random_seed = random_seed

        # LightGBM params
        self.lgb_params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
        }

        # Build features and targets
        self._prepare_data()

    def _prepare_data(self) -> None:
        # 1. Standardise US sector ETFs
        us_z = compute_rolling_z_scores_us(
            self.df_exec, self.us_tickers[:11], corr_window=60, ewma_half_life=45.0
        )
        for col in us_z.columns:
            self.df_exec[col] = us_z[col]

        # 2. Add TOPIX Close-to-Close return on sig_date as feature.
        # Since we don't have TOPIX Close-to-Close directly in df_exec, we use the equal-weighted
        # average of JP Close-to-Close returns at sig_date as a robust proxy for the market return,
        # or TOPIX if we can fetch it. Let's use the average of the 17 JP Close-to-Close returns
        # as the market return.
        jp_cc_cols = [f"jp_cc_{tk}" for tk in self.jp_tickers]
        self.df_exec["market_cc_sig"] = self.df_exec[jp_cc_cols].mean(axis=1)

        # 3. Calculate target variable: actual next day Close-to-Close return
        # r_{j, t+1} = (1 + jp_gap_{j, t+1}) * (1 + jp_oc_{j, t+1}) - 1
        for tk in self.jp_tickers:
            gap_col = f"jp_gap_{tk}"
            oc_col = f"jp_oc_{tk}"
            target_col = f"target_cc_{tk}"
            self.df_exec[target_col] = (1 + self.df_exec[gap_col]) * (1 + self.df_exec[oc_col]) - 1

        # 4. Calculate 20-day rolling volatility for target adjustment
        jp_vols = compute_jp_volatility(self.df_exec, self.jp_tickers, vol_window=20)
        for col in jp_vols.columns:
            self.df_exec[col] = jp_vols[col]

        # 5. Define Feature Columns
        self.feature_cols = list(us_z.columns) + ["market_cc_sig"]

    def run_rolling_predictions(
        self,
        start_date: str = "2020-01-01",
        vol_adjusted_target: bool = False,
    ) -> pd.DataFrame:
        """Run rolling walk-forward forecasting over the test period.

        Returns:
            DataFrame index=trade_date containing the predicted Close-to-Close returns
            for each JP ETF, columns are `pred_cc_{tk}`.
        """
        # Determine start index based on start_date
        start_dt = pd.to_datetime(start_date)
        start_idx = self.df_exec.index.get_indexer([start_dt], method="bfill")[0]

        # Ensure we have enough training data
        if start_idx < self.train_window:
            raise ValueError(
                f"Not enough history before {start_date} for train_window={self.train_window}. "
                f"First index is {start_idx}, need at least {self.train_window}."
            )

        T = len(self.df_exec)
        pred_returns = np.zeros((T, len(self.jp_tickers)))
        pred_returns[:] = np.nan

        # Keep a cache of the trained models for each asset
        models: Dict[str, LGBMPredictor] = {}

        # Features array
        X_all = self.df_exec[self.feature_cols].values

        for i in range(start_idx, T):
            trade_date = self.df_exec.index[i]
            
            # Determine if we should refit the models
            should_refit = ((i - start_idx) % self.refit_interval == 0)

            # Features for current step (sig_date t features)
            X_t = X_all[i].reshape(1, -1)

            # Training slice
            train_start = i - self.train_window
            train_end = i # excludes index i (i.e. up to i-1)
            
            X_train = X_all[train_start:train_end]

            # Train and predict for each ticker
            for j, tk in enumerate(self.jp_tickers):
                # Target preparation
                target_col = f"target_cc_{tk}"
                y_train_raw = self.df_exec[target_col].values[train_start:train_end]

                # Vol-adjustment
                if vol_adjusted_target:
                    # target z_{j, t+1} = r^{cc}_{j, t+1} / \sigma_{j, t}
                    # Note that \sigma_{j, t} is stored in row k of the DataFrame (vol_20_{tk})
                    # because row k has sig_date t.
                    vol_col = f"vol_20_{tk}"
                    vols_train = self.df_exec[vol_col].values[train_start:train_end]
                    
                    # Prevent division by zero
                    vols_train = np.maximum(vols_train, 1e-8)
                    y_train = y_train_raw / vols_train
                else:
                    y_train = y_train_raw

                # Clean any NaNs in train set (rare, but keep safe)
                valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
                X_tr_clean = X_train[valid_mask]
                y_tr_clean = y_train[valid_mask]

                if should_refit or tk not in models:
                    predictor = LGBMPredictor(
                        max_depth=self.lgb_params["max_depth"],
                        learning_rate=self.lgb_params["learning_rate"],
                        n_estimators=self.lgb_params["n_estimators"],
                        min_child_samples=self.lgb_params["min_child_samples"],
                        random_state=self.random_seed,
                    )
                    predictor.fit(X_tr_clean, y_tr_clean)
                    models[tk] = predictor
                else:
                    predictor = models[tk]

                # Predict
                pred_val = predictor.predict(X_t)[0]

                # Inverse transform if target was Z-score
                if vol_adjusted_target:
                    vol_col = f"vol_20_{tk}"
                    vol_t = self.df_exec[vol_col].values[i]
                    if not np.isfinite(vol_t) or vol_t <= 0:
                        vol_t = 1e-8
                    pred_returns[i, j] = pred_val * vol_t
                else:
                    pred_returns[i, j] = pred_val

        # Create results dataframe
        pred_cols = [f"pred_cc_{tk}" for tk in self.jp_tickers]
        pred_df = pd.DataFrame(pred_returns, index=self.df_exec.index, columns=pred_cols)
        return pred_df
