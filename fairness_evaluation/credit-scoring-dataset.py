import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI hangs
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import glob
import os
import argparse
import warnings

# Sklearn & Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Local imports
from compute_abroca import compute_abroca
from my_useful_functions import (
    calculate_performance_statistical_parity,
    calculate_performance_equalized_odds,
    calculate_performance_equal_opportunity,
    calculate_performance_predictive_parity,
    calculate_performance_predictive_equality,
    calculate_performance_treatment_equality,
    calculate_theil_index
)
from quality_metrics import prdc_score, detection_score, ks_score, tv_score, DCR

# ... (Previous imports remain, ensure `quality_metrics` is imported)

def run_eval_for_file(csv_path, model_subfolder, n_runs=5):
    # Load Original Data (Reference)
    # We assume the original data is at "data/credit-scoring.csv"
    # We need to load it as a raw dataframe for quality metrics
    original_csv_path = "data/credit-scoring.csv"
    if not os.path.exists(original_csv_path):
        print(f"Warning: Original data not found at {original_csv_path}. Skipping quality metrics.")
        df_real = None
    else:
        df_real = pd.read_csv(original_csv_path)
        # Impute NaNs for quality metrics (Data Level)
        cat_cols = df_real.select_dtypes(include=['object', 'string']).columns
        num_cols = df_real.select_dtypes(include=[np.number]).columns
        if len(cat_cols) > 0: df_real[cat_cols] = df_real[cat_cols].fillna("MISSING")
        if len(num_cols) > 0: df_real[num_cols] = df_real[num_cols].fillna(df_real[num_cols].median())
    
    # Load Synthetic Data for Quality Metrics
    try:
        df_fake = pd.read_csv(csv_path)
        if len(df_fake) < 10:
            print(f"Skipping {csv_path}: Too few samples ({len(df_fake)})")
            return

        # Impute NaNs for quality metrics (Data Level)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    cat_cols_f = df_fake.select_dtypes(include=['object', 'string']).columns
    num_cols_f = df_fake.select_dtypes(include=[np.number]).columns
    if len(cat_cols_f) > 0: df_fake[cat_cols_f] = df_fake[cat_cols_f].fillna("MISSING")
    if len(num_cols_f) > 0: df_fake[num_cols_f] = df_fake[num_cols_f].fillna(df_fake[num_cols_f].median())
    
    # --- Quality Metrics (Data Level) ---
    # We run this once per file (data-level), not per model/run.
    print(f"[DEBUG] Checking Quality Metrics: df_real exists? {df_real is not None}, model_subfolder='{model_subfolder}'")
    if df_real is not None and model_subfolder != "Original":

        print("-" * 50)
        print("QUALITY METRICS (Synthetic vs Original)")
        print("-" * 50)
        
        try:
            # PRDC
            prdc = prdc_score(df_real, df_fake)
            print(f"Precision: {prdc[0]:.4f}")
            print(f"Recall:    {prdc[1]:.4f}")
            
            val_density = prdc[2]
            if val_density < 0.1:
                print(f"Density:   {val_density*10:.4f}*")
            else:
                print(f"Density:   {val_density:.4f}")
                
            val_coverage = prdc[3]
            if val_coverage < 0.1:
                print(f"Coverage:  {val_coverage*10:.4f}*")
            else:
                print(f"Coverage:  {val_coverage:.4f}")
            
            # Detection Score
            det_score = detection_score(df_real, df_fake)
            print(f"Detection Score (ROC AUC): {det_score:.4f}")
            
            # DCR
            dcr_val = DCR(df_real, df_fake)
            print(f"DCR (Distance to Closest Record): {dcr_val[0]:.4f}")
            
            # KS & TV
            ks = ks_score(df_real, df_fake)
            tv = tv_score(df_real, df_fake)
            print(f"KS Score (Avg): {ks[0]:.4f}")
            print(f"TV Score (Avg): {tv[0]:.4f}")
            
            print("-" * 50)
        except Exception as e:
             print(f"Error calculating quality metrics: {e}")
             import traceback
             traceback.print_exc()

    # Continue with Fairness/Utility Evaluation
    try:
        X, y, sa_index, p_Group, protected_attribute, maj_name, min_name = load_credit_scoring_data(csv_path)
    except Exception as e:
        print(f"Skipping {csv_path}: {e}")
        return

    csv_name = os.path.basename(csv_path)
    print(f"\nModel: {model_subfolder} | Data: {csv_name}")
    print(f"Running {n_runs} evaluations...")
    
    algorithms = ['DT', 'MLP', 'kNN']
    results = {algo: [] for algo in algorithms}
    
    for i in range(n_runs):
        # Split Data with varying seed
        seed = 42 + i
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        # Scale Data (Important for MLP and kNN)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        for algo in algorithms:
            # Pass scaled data
            # Pass scaled data for training/prediction, but pass unscaled X_test for fairness evaluation
            metrics = run_experiment(model_subfolder, csv_name, X_train_scaled, X_test_scaled, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, maj_name, min_name, algo, run_id=i, verbose=True)
            results[algo].append(metrics)
            
    # Aggregate and Print Results
    metric_names = ['F1 Score', 'Statistical Parity', 'Equalized Odds', 'Predictive Parity', 
                    'Theil Index', 'ABROCA']
    
    print("-" * 100)
    print(f"{'Algorithm':<10} | {'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 100)
    
    for algo in algorithms:
        algo_metrics = results[algo]
        # Check if we have valid results
        if not algo_metrics:
            continue
            
        # For each metric, calculate mean and std
        for metric in metric_names:
            values = [m.get(metric, np.nan) for m in algo_metrics]
            # Filter NaNs/Infs if needed, but mean handles nan if robust
            values_clean = [v for v in values if pd.notna(v) and not np.isinf(v)]
            
            if values_clean:
                mean_val = np.mean(values_clean)
                std_val = np.std(values_clean)
                print(f"{algo:<10} | {metric:<20} | {mean_val:.4f}     | {std_val:.4f}")
            else:
                 print(f"{algo:<10} | {metric:<20} | NaN        | NaN")
        
        # Print LaTeX formatted string for this algo
        latex_metrics_order = ['F1 Score', 'Statistical Parity', 'Equalized Odds', 'Predictive Parity', 'ABROCA', 'Theil Index']
        latex_parts = []
        for metric in latex_metrics_order:
            values = [m.get(metric, np.nan) for m in algo_metrics]
            values_clean = [v for v in values if pd.notna(v) and not np.isinf(v)]
            if values_clean:
                mean_val = np.mean(values_clean)
                std_val = np.std(values_clean)
                latex_parts.append(f"${mean_val:.4f}_{{\\,\\pm {std_val:.4f}}}$")
            else:
                 latex_parts.append(f"NaN_{{\\,\\pm NaN}}")
        
        print("Latex: " + " & ".join(latex_parts) + " \\\\")
        print("-" * 100)
try:
    from pomegranate import *
    import pygraphviz
    from pomegranate.utils import plot_networkx
    POMEGRANATE_INSTALLED = True
except ImportError:
    POMEGRANATE_INSTALLED = False

import networkx

warnings.filterwarnings('ignore')

# Set root directories
SYNTHETIC_ROOT = "/Users/doanthang/Desktop/Code/Fairness/Dist_Learn_Fair_Tab/synthetic_data/credit-scoring"
BASELINE_ROOT = "baseline_data" # Assumed relative to project root
EVAL_OUTPUT_ROOT = "fairness_evaluation/credit-scoring"

def get_bn_structure_and_data(csv_path):
    """
    Loads data for BN plotting.
    Returns: a (structure if known, else None), X (data), labels (column names)
    """
    df = pd.read_csv(csv_path)
    
    df_encoded = df.copy()
    label_encoders = {}
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            label_encoders[column] = le
            
    X = df_encoded.values
    labels = df.columns.tolist()
    return None, X, labels

def plot_bayesian_network(a, X, labels, save_path):
    try:
        model = BayesianNetwork.from_samples(X, algorithm='greedy', constraint_graph=a, state_names=labels)
        plt.figure(figsize=(20, 16), dpi=400)
        model.plot()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved BN plot to {save_path}")
    except Exception as e:
        print(f"Failed to plot BN: {e}")

def load_credit_scoring_data(csv_path):
    """
    Loads Credit Scoring dataset, encodes categorical features, 
    identifies protected attribute (sex) and label (class-label).
    
    Mappings:
    - Sex: 1=Male (Privileged), 2=Female (Unprivileged) -> Map to 1/0
    - Label: 0=Approved (Positive), 1=Rejected (Negative) -> Map to 1/0
    """
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    
    # Identify Target and Protected Attribute
    # Note: Column names might vary slightly (case sensitivity), so we normalize if needed
    # But synthetic data generator usually produces consistent headers.
    
    # Check for 'Sex' or 'sex'
    protected_attr_col = 'Sex'
    if 'Sex' not in df.columns and 'sex' in df.columns:
        protected_attr_col = 'sex'
        
    target_col = 'class-label'
    if target_col not in df.columns and 'class_label' in df.columns:
        target_col = 'class_label'
    
    if target_col not in df.columns or protected_attr_col not in df.columns:
        # Try to find them dynamically
        cols_lower = {c.lower(): c for c in df.columns}
        if 'sex' in cols_lower: protected_attr_col = cols_lower['sex']
        if 'class-label' in cols_lower: target_col = cols_lower['class-label']
        elif 'class_label' in cols_lower: target_col = cols_lower['class_label']
        elif 'label' in cols_lower: target_col = cols_lower['label']
        
    if target_col not in df.columns or protected_attr_col not in df.columns:
         raise ValueError(f"CSV must contain target and protected attribute columns. Found: {df.columns.tolist()}")

    # --- Preprocessing ---

    # 1. Encode Protected Attribute (Sex)
    # Original: 1 = Male, 2 = Female
    # Desired: 1 = Male (Privileged), 0 = Female (Unprivileged)
    
    # First ensure it's numeric
    df[protected_attr_col] = pd.to_numeric(df[protected_attr_col], errors='coerce').fillna(0).astype(int)
    
    # Map: 1 -> 1, 2 -> 0. (Any other values -> 0 to be safe)
    # If the synthetic data outputs 1 and 2:
    if df[protected_attr_col].max() > 1:
        # Assuming 1 is Male, 2 is Female based on standard Credit Scoring dataset from UCI/Statlog often having this.
        # But wait, verify standard Statlog (German Credit) vs "Credit Scoring" (maybe custom).
        # In the 'data/credit-scoring.csv', we saw 'Sex' values like 1, 2.
        # We will map 1 -> 1 (Male) and 2 -> 0 (Female).
        df[protected_attr_col] = df[protected_attr_col].apply(lambda x: 1 if x == 1 else 0)
    
    # 2. Encode Target (class-label)
    # Original: 0 = Approved, 1 = Rejected (Based on assumption and standard practice where 0 is often 'good' in some raw formats, or check variable desc)
    # User confirmed: "0=Approved (Positive Outcome)".
    # Desired: 1 = Approved (Positive), 0 = Rejected (Negative) for metric calculations
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(1).astype(int)
    
    # Map: 0 -> 1, 1 -> 0
    df[target_col] = 1 - df[target_col]
    
    # 3. Handle other categorical features
    le_dict = {}
    for col in df.columns:
        if col == target_col or col == protected_attr_col:
            continue
            
        # Check if column is not numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            
    # X and y
    # Drop target and any score-related columns that leak the label
    drop_cols = [target_col]
    
    # Potential leakage columns based on data exploration
    leakage_candidates = ['Score-level', 'Score-class', 'Score_point', 'score-level', 'score-class', 'score_point', 'INPS-yes-no', 'inps-yes-no']
    
    for leak in leakage_candidates:
        # Check case-insensitive
        cols_lower = {c.lower(): c for c in df.columns}
        if leak.lower() in cols_lower:
            drop_cols.append(cols_lower[leak.lower()])
            
    # Also drop columns that might be ID or not useful features if needed, but for now focus on leakage
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    print(f"Features used for training: {X.columns.tolist()}")

    # Identify index of protected attribute in X
    sa_index = list(X.columns).index(protected_attr_col)
    
    # Privileged Group Value (Male=1)
    p_Group = 1
    
    return X, y, sa_index, p_Group, protected_attr_col, "Male", "Female"

def run_experiment(model_subfolder, csv_name, X_train, X_test, X_test_orig, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, algorithm, run_id=0, verbose=False):
    # Train Model
    if algorithm == 'DT':
        clf = DecisionTreeClassifier(random_state=run_id)
    elif algorithm == 'NB':
        clf = GaussianNB()
    elif algorithm == 'MLP':
         clf = MLPClassifier(random_state=run_id, max_iter=300)
    elif algorithm == 'kNN':
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        return {}

    
        
    clf.fit(X_train, y_train)
    
        
    y_predicts = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    if verbose:
        print(f"  [DEBUG] Finished {algorithm}. Calculating metrics...")
    
    metrics_dict = {}
    
    # Fairness Metrics
    try:
        # Use X_test_orig for fairness metrics to ensure sensitive attribute values match saValue (e.g. 0/1)
        sp_res = calculate_performance_statistical_parity(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)
        
        metrics_dict['Accuracy'] = sp_res['accuracy']
        metrics_dict['F1 Score'] = sp_res['f1-score']  # Explicitly add F1 Score to dict
        
        if np.isnan(metrics_dict['F1 Score']):
             print(f"[DEBUG] F1 is NaN for {algorithm}. y_test unique: {np.unique(y_test)}, y_pred unique: {np.unique(y_predicts)}")
        
        metrics_dict['Balanced Accuracy'] = sp_res['balanced_accuracy']
        
        metrics_dict['Statistical Parity'] = sp_res['fairness']
        metrics_dict['Equalized Odds'] = calculate_performance_equalized_odds(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        # metrics_dict['Equal Opportunity'] = calculate_performance_equal_opportunity(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        metrics_dict['Predictive Parity'] = calculate_performance_predictive_parity(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        metrics_dict['Predictive Equality'] = calculate_performance_predictive_equality(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        # metrics_dict['Treatment Equality'] = calculate_performance_treatment_equality(X_test_orig.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']

        
        # Individual Fairness: Theil Index
        metrics_dict['Theil Index'] = calculate_theil_index(y_probs)

        # ABROCA
        # Need dataframe for X_test with proper column names for ABROCA function
        X_test_df = X_test_orig.copy()
        X_test_df['label'] = y_test.values
        X_test_df['pred'] = y_probs
        
        # Plot ABROCA (Only for the first run to avoid overhead/overwriting)
        if run_id == 0:
            save_file = os.path.join(EVAL_OUTPUT_ROOT, model_subfolder, f"{algorithm}_{csv_name}.abroca.pdf")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
            # Pass integer 1 for majority group and 'Sex' (or whatever name) for protected col
            abroca_score = compute_abroca(
                X_test_df, 
                pred_col='pred', 
                label_col='label', 
                protected_attr_col=protected_attribute, 
                majority_protected_attr_val=1,
                file_name=save_file,
                plot_slices=True
            )
        else:
            # Just compute score without plotting
             abroca_score = compute_abroca(
                X_test_df, 
                pred_col='pred', 
                label_col='label', 
                protected_attr_col=protected_attribute, 
                majority_protected_attr_val=1,
                file_name=None,
                plot_slices=False
            )
        metrics_dict['ABROCA'] = abroca_score
        
    except Exception as e:
        print(f"Error calculating metrics for {algorithm}: {e}")
        import traceback
        traceback.print_exc()
        
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate fairness for synthetic Credit Scoring datasets.")
    parser.add_argument("--plot-bn", action="store_true", help="Enable Bayesian Network plotting.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for evaluation (default: 5).")
    parser.add_argument("--original", action="store_true", help="Also evaluate original dataset.")
    args = parser.parse_args()

    # 1. Evaluate Original Dataset if requested
    if args.original:
        original_data_path = "data/credit-scoring.csv"
        if os.path.exists(original_data_path):
            print("\n" + "="*50)
            print("EVALUATING ORIGINAL DATASET")
            print("="*50)
            # Use "Original" as model_subfolder so results go to fairness_evaluation/credit-scoring/Original/
            run_eval_for_file(original_data_path, "Original", n_runs=args.runs)
        else:
            print(f"Model original requested but {original_data_path} not found.")

    # 2. Evaluate Synthetic Datasets
    # Find all CSV files in subdirectories
    search_pattern = os.path.join(SYNTHETIC_ROOT, "*", "*.csv")
    csv_files = glob.glob(search_pattern)
    print(f"\nFound {len(csv_files)} synthetic datasets in {SYNTHETIC_ROOT}")
    
    for csv_path in csv_files:
        # Determine model name from parent directory
        parent_dir = os.path.dirname(csv_path)
        model_subfolder = os.path.basename(parent_dir)
        
        print(f"\nProcessing: {model_subfolder} / {os.path.basename(csv_path)}")

        # Run BN Plot (Optional) - only for the file, not repeated runs
        if args.plot_bn:
            if POMEGRANATE_INSTALLED:
                save_path = os.path.join(EVAL_OUTPUT_ROOT, model_subfolder, f"BN_{os.path.basename(csv_path)}.pdf")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                try:
                    a, X, labels = get_bn_structure_and_data(csv_path)
                    plot_bayesian_network(a, X, labels, save_path)
                except Exception as e:
                    print(f"Error preparing BN data/plot: {e}")
            else:
                print("Warning: --plot-bn requested but 'pomegranate' not installed. Skipping BN plot.")
        
        # Run Evaluation
        run_eval_for_file(csv_path, model_subfolder, n_runs=args.runs)

    # 3. Evaluate Baseline Datasets
    # Find all CSV files in subdirectories of baseline_data matching the dataset name
    # credits scoring filename: credit-scoring.csv
    search_pattern_baseline = os.path.join(BASELINE_ROOT, "*", "credit-scoring.csv")
    baseline_files = glob.glob(search_pattern_baseline)
    
    if baseline_files:
        print("\n" + "="*50)
        print("EVALUATING BASELINE DATASETS")
        print("="*50)
        print(f"Found {len(baseline_files)} baseline datasets in {BASELINE_ROOT}")
        
        for csv_path in baseline_files:
            # Determine method name from parent directory (e.g. 'ctgan', 'tabgen')
            parent_dir = os.path.dirname(csv_path)
            method_name = os.path.basename(parent_dir)
            model_subfolder = f"Baseline_{method_name}"
            
            print(f"\nProcessing Baseline: {method_name} / {os.path.basename(csv_path)}")
            
            # Run Evaluation
            run_eval_for_file(csv_path, model_subfolder, n_runs=args.runs)

if __name__ == "__main__":
    main()
