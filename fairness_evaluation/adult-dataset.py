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

# Conditional imports for Pomegranate
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
SYNTHETIC_ROOT = "/Users/doanthang/Desktop/Code/Fairness/Dist_Learn_Fair_Tab/synthetic_data/adult"
BASELINE_ROOT = "baseline_data" # Assumed relative to project root
EVAL_OUTPUT_ROOT = "fairness_evaluation/adult"

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
        # Pomegranate BN plotting
        model = BayesianNetwork.from_samples(X, algorithm='greedy', constraint_graph=a, state_names=labels)
        plt.figure(figsize=(20, 16), dpi=400)
        model.plot()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved BN plot to {save_path}")
    except Exception as e:
        print(f"Failed to plot BN: {e}")

def load_adult_data(csv_path):
    """
    Loads Adult dataset, encodes categorical features, 
    identifies protected attribute (gender) and label (class-label).
    """
    df = pd.read_csv(csv_path)
    # Drop missing values to handle potential NaN issues
    df.dropna(inplace=True)
    
    # Preprocessing
    # 1. Identify Target and Protected Attribute
    target_col = 'class-label'
    protected_attr_col = 'gender'
    
    # Verify columns exist
    if target_col not in df.columns:
         # Try variants
         if 'class_label' in df.columns: target_col = 'class_label'
         elif 'income' in df.columns: target_col = 'income'

    if protected_attr_col not in df.columns:
         if 'sex' in df.columns: protected_attr_col = 'sex'

    if target_col not in df.columns or protected_attr_col not in df.columns:
        # If still not found, try to guess or hardcode based on column index if needed, 
        # but better to raise error or return defaults.
        # Check for column names in lowercase
        cols_lower = {c.lower(): c for c in df.columns}
        if 'gender' in cols_lower: protected_attr_col = cols_lower['gender']
        if 'sex' in cols_lower: protected_attr_col = cols_lower['sex']
        if 'class-label' in cols_lower: target_col = cols_lower['class-label']
        if 'income' in cols_lower: target_col = cols_lower['income']

    if target_col not in df.columns or protected_attr_col not in df.columns:
        # Fallback: assume last column is target 
         target_col = df.columns[-1]
         # Assume 'sex' or 'gender' is present
         
    # 2. Encode Protected Attribute 
    # 'Male' -> 1 (Privileged), 'Female' -> 0 (Unprivileged) usually
    if df[protected_attr_col].dtype == 'object':
         # Map explicit values first if known
         gender_map = {'Male': 1, 'Female': 0, ' Male': 1, ' Female': 0, 'm': 1, 'f': 0}
         # For any values not in map, standard LabelEncoder might be safer but we want specific direction
         # Apply map, fill others with 0
         # But wait, LabelEncoder assigns 1 to Male (alphabetically F comes before M? No, F=0, M=1. Yes.)
         le_sex = LabelEncoder()
         df[protected_attr_col] = le_sex.fit_transform(df[protected_attr_col].astype(str))
         # Check if 'Male' became 1. If 'Female' (F) and 'Male' (M), F->0, M->1. Correct.
         
    # Ensure int
    df[protected_attr_col] = pd.to_numeric(df[protected_attr_col], errors='coerce').fillna(0).astype(int)

    # 3. Encode Target
    # '1' -> >50K, '0' -> <=50K. 
    if df[target_col].dtype == 'object':
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str))
        # Usually <=50K (starts with <) comes before >50K. 
        # So 0: <=50K, 1: >50K.
        # Verify: Metric usually expects 1 as Positive (Good/High income).
    
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
    
    # 4. Handle other categorical features
    le_dict = {}
    for col in df.columns:
        if col == target_col or col == protected_attr_col:
             continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            
    # X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify index of protected attribute in X
    sa_index = list(X.columns).index(protected_attr_col)
    
    # Privileged Group Value (Male=1)
    p_Group = 1
    
    return X, y, sa_index, p_Group, protected_attr_col, "Male", "Female"

def run_experiment(model_subfolder, csv_name, X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, majority_group_name, minority_group_name, algorithm, run_id=0, verbose=False):
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
        print(f"--- Results for {algorithm} on {csv_name} ---")
    
    metrics_dict = {}
    
    # Fairness Metrics
    try:
        sp_res = calculate_performance_statistical_parity(X_test.values, y_test.values, y_predicts, sa_index, p_Group)
        
        metrics_dict['F1 Score'] = sp_res['f1-score']
        
        metrics_dict['Statistical Parity'] = sp_res['fairness']
        metrics_dict['Equalized Odds'] = calculate_performance_equalized_odds(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        metrics_dict['Predictive Parity'] = calculate_performance_predictive_parity(X_test.values, y_test.values, y_predicts, sa_index, p_Group)['fairness']
        
        # Individual Fairness
        metrics_dict['Theil Index'] = calculate_theil_index(y_probs)

        # ABROCA
        # Need dataframe for X_test with proper column names for ABROCA function
        X_test_df = X_test.copy()
        X_test_df['label'] = y_test.values
        X_test_df['pred'] = y_probs
        
        # Plot ABROCA (Only first run)
        if run_id == 0:
            save_file = os.path.join(EVAL_OUTPUT_ROOT, model_subfolder, f"{algorithm}_{csv_name}.abroca.pdf")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
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

def run_eval_for_file(csv_path, model_subfolder, n_runs=5):
    # Load Original Data (Reference) for Quality Metrics
    original_csv_path = "data/adult.csv"
    if not os.path.exists(original_csv_path):
         print(f"Warning: Original data not found at {original_csv_path}. Skipping quality metrics.")
         df_real = None
    else:
         df_real = pd.read_csv(original_csv_path)
         # Impute NaNs for quality metrics
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

        # Impute NaNs for quality metrics
        cat_cols_f = df_fake.select_dtypes(include=['object', 'string']).columns
        num_cols_f = df_fake.select_dtypes(include=[np.number]).columns
        if len(cat_cols_f) > 0: df_fake[cat_cols_f] = df_fake[cat_cols_f].fillna("MISSING")
        if len(num_cols_f) > 0: df_fake[num_cols_f] = df_fake[num_cols_f].fillna(df_fake[num_cols_f].median())
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    # --- Quality Metrics (Data Level) ---
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
            
            # DCR (Sampled if necessary)
            if len(df_real) > 2000:
                df_real_sample = df_real.sample(n=2000, random_state=42)
            else:
                df_real_sample = df_real
                
            if len(df_fake) > 2000:
                 df_fake_sample = df_fake.sample(n=2000, random_state=42)
            else:
                 df_fake_sample = df_fake
                 
            try:
                dcr_val = DCR(df_real_sample, df_fake_sample)
                print(f"DCR (Distance to Closest Record): {dcr_val[0]:.4f}")
            except Exception as e:
                print(f"Error calculating DCR: {e}")
            
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

    # Load Data for Fairness Eval
    try:
        X, y, sa_index, p_Group, protected_attribute, maj_name, min_name = load_adult_data(csv_path)
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
        except ValueError:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
        
        for algo in algorithms:
            metrics = run_experiment(model_subfolder, csv_name, X_train, X_test, y_train, y_test, sa_index, p_Group, protected_attribute, maj_name, min_name, algo, run_id=i)
            results[algo].append(metrics)
            
    # Aggregate and Print Results
    metric_names = ['F1 Score', 'Statistical Parity', 'Equalized Odds', 'Predictive Parity', 
                    'Theil Index', 'ABROCA']
    
    print("-" * 100)
    print(f"{'Algorithm':<10} | {'Metric':<20} | {'Mean':<10} | {'Std':<10}")
    print("-" * 100)
    
    for algo in algorithms:
        algo_metrics = results[algo]
        if not algo_metrics: continue
            
        for metric in metric_names:
            values = [m.get(metric, np.nan) for m in algo_metrics]
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate fairness for synthetic Adult datasets.")
    parser.add_argument("--plot-bn", action="store_true", help="Enable Bayesian Network plotting.")
    parser.add_argument("--runs", type=int, default=5, help="Number of evaluation runs.")
    parser.add_argument("--original", action="store_true", help="Evaluate original dataset.")
    args = parser.parse_args()

    # 1. Evaluate Original Dataset if requested
    if args.original:
        original_data_path = "data/adult.csv"
        if os.path.exists(original_data_path):
            print("\n" + "="*50)
            print("EVALUATING ORIGINAL DATASET")
            print("="*50)
            run_eval_for_file(original_data_path, "Original", n_runs=args.runs)
        else:
            print(f"Model original requested but {original_data_path} not found.")

    # 2. Evaluate Synthetic Datasets
    search_pattern = os.path.join(SYNTHETIC_ROOT, "*", "*.csv")
    csv_files = glob.glob(search_pattern)
    print(f"\nFound {len(csv_files)} synthetic datasets in {SYNTHETIC_ROOT}")
    
    for csv_path in csv_files:
        parent_dir = os.path.dirname(csv_path)
        model_subfolder = os.path.basename(parent_dir)
        
        print(f"\nProcessing: {model_subfolder} / {os.path.basename(csv_path)}")

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
    # adult filename: adult.csv
    search_pattern_baseline = os.path.join(BASELINE_ROOT, "*", "adult.csv")
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
