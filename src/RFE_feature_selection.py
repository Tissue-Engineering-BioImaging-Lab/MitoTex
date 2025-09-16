# # RFE_feature_selection.py
# import pandas as pd
# import numpy as np
# import os
# import traceback
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE
# from sklearn.ensemble import RandomForestClassifier

# def run_rfe_feature_selection(
#     input_csv,
#     output_dir,
#     top_k=20,
#     n_estimators=200,
#     n_jobs=-1,
#     save_summary=True
# ):
#     """
#     Performs RFE with Random Forest on the input dataset.
#     Returns:
#       summary_df, reduced_csv_path, summary_csv_path (or None)
#     """
#     try:
#         # --- Load dataset ---
#         df = pd.read_csv(input_csv)

#         # keep only numeric columns (features + target)
#         numeric_df = df.select_dtypes(include=[np.number])
#         if numeric_df.shape[1] < 2:
#             raise ValueError("Need at least one feature column and one target column (numeric).")

#         feature_columns = numeric_df.columns[:-1]  # features (names)
#         target_column = numeric_df.columns[-1]     # label name

#         # Impute missing values (features only) and keep original rows aligned
#         imputer = SimpleImputer(strategy="mean")
#         X_imputed = imputer.fit_transform(numeric_df[feature_columns])
#         y = numeric_df[target_column].values

#         # Drop rows with missing labels (if any)
#         valid_mask = ~pd.isna(y)
#         if not valid_mask.all():
#             X_imputed = X_imputed[valid_mask]
#             y = y[valid_mask]

#         # Convert y to integer labels when appropriate
#         try:
#             y = y.astype(int)
#         except Exception:
#             # if casting to int fails keep original (e.g. categorical encoded as other)
#             pass

#         # Standardize features for RFE (but keep imputed raw X for saving reduced CSV)
#         X_std = StandardScaler().fit_transform(X_imputed)

#         n_features = X_std.shape[1]
#         if top_k > n_features:
#             top_k = n_features

#         # --- Random Forest estimator used inside RFE ---
#         rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)

#         # --- RFE ---
#         rfe = RFE(estimator=rf, n_features_to_select=top_k, step=1)
#         rfe.fit(X_std, y)

#         support = rfe.support_                      # boolean mask (length = n_features)
#         ranking = rfe.ranking_                      # ranks for each original feature
#         selected_idx = np.where(support)[0]         # indices of selected features
#         selected_features = feature_columns[support]  # names

#         # The final estimator inside RFE is fitted to the selected features.
#         # Its feature_importances_ length == top_k, so map them back to the selected feature names.
#         final_importances = rfe.estimator_.feature_importances_
#         # Build a Series aligned with all original features (NaN for unselected features)
#         importances_full = pd.Series(data=np.nan, index=feature_columns)
#         importances_full.iloc[selected_idx] = final_importances

#         # Stats for selected features (use imputed but unscaled X for interpretable values)
#         mean_vals = np.mean(X_imputed[:, selected_idx], axis=0)
#         std_vals  = np.std(X_imputed[:, selected_idx], axis=0)
#         min_vals  = np.min(X_imputed[:, selected_idx], axis=0)
#         max_vals  = np.max(X_imputed[:, selected_idx], axis=0)

#         summary_df = pd.DataFrame({
#             "Feature": feature_columns,
#             "Rank": ranking,
#             "Selected": support,
#             "Importance": importances_full.values
#         })

#         # Add per-selected-feature stats (for selected ones only)
#         stats_df = pd.DataFrame({
#             "Feature": selected_features,
#             "Mean": mean_vals,
#             "Std": std_vals,
#             "Min": min_vals,
#             "Max": max_vals
#         })

#         # Merge summary_df with stats (stats will be NaN for non-selected)
#         summary_df = summary_df.merge(stats_df, on="Feature", how="left")
#         # Sort by rank (best rank = 1 at top)
#         summary_df = summary_df.sort_values(by="Rank").reset_index(drop=True)

#         # --- Save reduced dataset (imputed raw values, selected features) ---
#         os.makedirs(output_dir, exist_ok=True)
#         base_name = os.path.splitext(os.path.basename(input_csv))[0]
#         reduced_path = os.path.join(output_dir, f"Top_{top_k}_RFE_{base_name}.csv")
#         reduced_df = pd.DataFrame(X_imputed[:, selected_idx], columns=selected_features)
#         reduced_df[target_column] = y
#         reduced_df.to_csv(reduced_path, index=False)

#         summary_path = None
#         if save_summary:
#             summary_path = os.path.join(output_dir, f"RFE_summary_{base_name}.csv")
#             summary_df.to_csv(summary_path, index=False)

#         return summary_df, reduced_path, summary_path

#     except Exception:
#         tb = traceback.format_exc()
#         raise RuntimeError(f"RFE failed: {tb}")
# RFE_feature_selection.py
import pandas as pd
import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def run_rfe_feature_selection(
    input_csv,
    output_dir,
    top_k=20,
    n_estimators=200,
    n_jobs=-1,
    save_summary=True
):
    """
    Performs RFE with Random Forest on the input dataset.
    Returns:
      summary_df, reduced_csv_path, summary_csv_path (or None)
    Also plots:
      - RFE ranking of all features
      - Feature importances of selected features
    """
    try:
        # --- Load dataset ---
        df = pd.read_csv(input_csv)

        # keep only numeric columns (features + target)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least one feature column and one target column (numeric).")

        feature_columns = numeric_df.columns[:-1]  # features (names)
        target_column = numeric_df.columns[-1]     # label name

        # Impute missing values (features only)
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(numeric_df[feature_columns])
        y = numeric_df[target_column].values

        # Drop rows with missing labels (if any)
        valid_mask = ~pd.isna(y)
        if not valid_mask.all():
            X_imputed = X_imputed[valid_mask]
            y = y[valid_mask]

        # Convert y to integer labels when appropriate
        try:
            y = y.astype(int)
        except Exception:
            pass

        # Standardize features for RFE
        X_std = StandardScaler().fit_transform(X_imputed)
        n_features = X_std.shape[1]
        if top_k > n_features:
            top_k = n_features

        # --- Random Forest estimator used inside RFE ---
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs)

        # --- RFE ---
        rfe = RFE(estimator=rf, n_features_to_select=top_k, step=1)
        rfe.fit(X_std, y)

        support = rfe.support_                      # boolean mask (length = n_features)
        ranking = rfe.ranking_                      # ranks for each original feature
        selected_idx = np.where(support)[0]         # indices of selected features
        selected_features = feature_columns[support]  # names

        # Feature importances from RFE estimator
        final_importances = rfe.estimator_.feature_importances_
        importances_full = pd.Series(data=np.nan, index=feature_columns)
        importances_full.iloc[selected_idx] = final_importances

        # Stats for selected features
        mean_vals = np.mean(X_imputed[:, selected_idx], axis=0)
        std_vals  = np.std(X_imputed[:, selected_idx], axis=0)
        min_vals  = np.min(X_imputed[:, selected_idx], axis=0)
        max_vals  = np.max(X_imputed[:, selected_idx], axis=0)

        summary_df = pd.DataFrame({
            "Feature": feature_columns,
            "Rank": ranking,
            "Selected": support,
            "Importance": importances_full.values
        })

        stats_df = pd.DataFrame({
            "Feature": selected_features,
            "Mean": mean_vals,
            "Std": std_vals,
            "Min": min_vals,
            "Max": max_vals
        })

        summary_df = summary_df.merge(stats_df, on="Feature", how="left")
        summary_df = summary_df.sort_values(by="Rank").reset_index(drop=True)

        # --- Save reduced dataset ---
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        reduced_path = os.path.join(output_dir, f"Top_{top_k}_RFE_{base_name}.csv")
        reduced_df = pd.DataFrame(X_imputed[:, selected_idx], columns=selected_features)
        reduced_df[target_column] = y
        reduced_df.to_csv(reduced_path, index=False)

        summary_path = None
        if save_summary:
            summary_path = os.path.join(output_dir, f"RFE_summary_{base_name}.csv")
            summary_df.to_csv(summary_path, index=False)

        # --- PLOTTING ---
        # RFE ranking plot
        plt.figure(figsize=(12,6))
        plt.bar(feature_columns, ranking)
        plt.xticks(rotation=90)
        plt.title("RFE Feature Ranking (1 = Best)")
        plt.xlabel("Feature")
        plt.ylabel("Rank")
        plt.tight_layout()
        plt.show()

        # Feature importance plot
        plt.figure(figsize=(10,6))
        plt.bar(selected_features, final_importances)
        plt.xticks(rotation=90)
        plt.title("Random Forest Importances (Top Features via RFE)")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()

        return summary_df, reduced_path, summary_path

    except Exception:
        tb = traceback.format_exc()
        raise RuntimeError(f"RFE failed: {tb}")
