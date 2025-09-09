# RFE_feature_selection.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import os
import traceback

def run_rfe_feature_selection(
    input_csv,
    output_dir,
    top_k=20,
    rf_only=False,
    n_estimators=100,
    n_jobs=1,
    step=None,
    save_summary=True
):
    """
    Returns:
      summary_df, reduced_csv_path, summary_csv_path (or None)
    """
    try:
        # --- load & basic checks ---
        df = pd.read_csv(input_csv)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least one feature column and one target column (numeric).")

        feature_columns = list(numeric_df.columns[:-1])  # features
        target_column = numeric_df.columns[-1]           # last numeric column assumed target

        # --- impute features ---
        X = SimpleImputer(strategy='mean').fit_transform(numeric_df[feature_columns])

        # --- handle labels (drop NaNs, keep values intact) ---
        y = numeric_df[target_column].values
        valid_mask = ~pd.isna(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(int)  # force to int if numeric

        if X.size == 0:
            raise ValueError("No valid feature rows after cleaning.")

        # --- standardize ---
        X_std = StandardScaler().fit_transform(X)
        n_features = X_std.shape[1]

        # make sure top_k is valid
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > n_features:
            top_k = n_features

        # choose sensible integer step
        if step is None:
            step = 1 if n_features <= 50 else max(1, n_features // 10)

        # --- prepare model ---
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)

        # --- RF-only path ---
        if rf_only:
            model.fit(X_std, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_k]
            selected_mask = np.zeros(n_features, dtype=bool)
            selected_mask[indices] = True

            selected_features = [feature_columns[i] for i in indices]
            final_importances = importances[indices]
            means = X_std[:, indices].mean(axis=0)
            stds = X_std[:, indices].std(axis=0)
            mins = X_std[:, indices].min(axis=0)
            maxs = X_std[:, indices].max(axis=0)

            summary = pd.DataFrame({
                "Feature": selected_features,
                "Importance": final_importances,
                "Mean": means,
                "Std": stds,
                "Min": mins,
                "Max": maxs
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            reduced_df = pd.DataFrame(X[:, indices], columns=selected_features)
            reduced_df[target_column] = y
        else:
            # --- RFE path ---
            rfe = RFE(estimator=model, n_features_to_select=top_k, step=step)
            rfe.fit(X_std, y)

            support = rfe.support_
            if not hasattr(rfe, "estimator_"):
                raise RuntimeError("RFE did not produce a fitted estimator_.")
            final_model = rfe.estimator_
            final_importances = getattr(final_model, "feature_importances_", None)

            selected_idx = np.where(support)[0]
            selected_features = [feature_columns[i] for i in selected_idx]

            if final_importances is None:
                tmp_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)
                tmp_clf.fit(X_std[:, selected_idx], y)
                final_importances = tmp_clf.feature_importances_

            means = X_std[:, selected_idx].mean(axis=0)
            stds = X_std[:, selected_idx].std(axis=0)
            mins = X_std[:, selected_idx].min(axis=0)
            maxs = X_std[:, selected_idx].max(axis=0)

            summary = pd.DataFrame({
                "Feature": selected_features,
                "Importance": final_importances,
                "Mean": means,
                "Std": stds,
                "Min": mins,
                "Max": maxs
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            reduced_df = pd.DataFrame(X[:, selected_idx], columns=selected_features)
            reduced_df[target_column] = y

        # --- save results ---
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        reduced_path = os.path.join(output_dir, f"Top_{top_k}_RFE_features.csv")
        reduced_df.to_csv(reduced_path, index=False)

        summary_path = None
        if save_summary:
            summary_path = os.path.join(output_dir, f"RFE_summary_{base_name}.csv")
            summary.to_csv(summary_path, index=False)

        return summary, reduced_path, summary_path

    except Exception:
        tb = traceback.format_exc()
        raise RuntimeError(f"RFE failed: {tb}")
