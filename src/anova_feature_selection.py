import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
import os

def run_anova_filter(input_csv, top_k=10, output_dir=None, plot=True, save_output=True):
    
    # Load dataset
    data = pd.read_csv(input_csv)
    numeric_data = data.select_dtypes(include=[np.number])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_data = imputer.fit_transform(numeric_data)

    # Split into X and Y
    X = numeric_data[:, :-1]
    Y = numeric_data[:, -1]

    # Standardize features
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Feature selection
    fs = SelectKBest(score_func=f_classif, k=top_k)
    fs.fit(X_std, Y)
    X_fs = fs.transform(X_std)

    # Sort scores and select top_k
    structuredScores = np.array([[i, score] for i, score in enumerate(fs.scores_)])
    structuredScores = structuredScores[structuredScores[:, 1].argsort()[::-1]][:top_k]
    structuredScores = structuredScores[structuredScores[:, 0].argsort()]

    if plot:
        plt.bar(range(len(fs.scores_)), fs.scores_)
        plt.title("Feature Univariate ANOVA Scores")
        plt.xlabel("Feature Index")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.show()

    # Calculate stats
    means = np.mean(X_fs, axis=0)
    stds = np.std(X_fs, axis=0)
    mins = np.min(X_fs, axis=0)
    maxs = np.max(X_fs, axis=0)

    finalOutput = np.column_stack((structuredScores, means, stds, mins, maxs))
    df = pd.DataFrame(finalOutput, columns=['Feature', 'Score', 'Mean', 'Std', 'Min', 'Max'])
    df['Feature'] = df['Feature'].astype(int)

    out_path = None
    if save_output:
        selected_indices = df['Feature'].values.astype(int)
        selected_columns = data.select_dtypes(include=[np.number]).columns[selected_indices]
        filtered_data = data[selected_columns]
        filtered_data.loc[:, 'Last_Column'] = data.iloc[:, -1]


        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        out_filename = f"Anova_{base_name}.csv"
        out_path = os.path.join(output_dir or os.path.dirname(input_csv), out_filename)
        filtered_data.to_csv(out_path, index=False)

    return df, out_path
