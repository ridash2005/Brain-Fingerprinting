# %% [markdown]
# # MetricBolT Baseline - Brain Fingerprinting Comparison
# 
# This notebook implements the MetricBolT (Metric Learning + BolT Transformer) 
# baseline for direct comparison against our ConvAE+SDL Brain Fingerprinting pipeline.
#
# **Key Design Principles (Awareness-Compliant):**
# - 5-Fold Cross-Validation with strict subject-level splits
# - Model trained on TRAIN subjects only, evaluated on HELD-OUT subjects
# - Prevents identifiability leakage (Orlichenko et al., 2023)
#
# **Evaluation Protocol:**
# - Identical metrics to `kaggle_brain_fingerprinting.py`
# - Same data loading, same BOLD naming, same subject intersection
# - Results directly comparable
#
# ---

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
import sys
import subprocess
import json
import shutil
import datetime
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "metricbolt_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ==========================================
# CLONE METRICBOLT & INSTALL DEPENDENCIES
# ==========================================
METRICBOLT_REPO_URL = "https://github.com/CAIMI-WEIGroup/MetricBolT.git"
REPO_DIR = os.path.join(WORKING_DIR, "MetricBolT")

if not os.path.exists(REPO_DIR):
    print(f"[*] Cloning MetricBolT from {METRICBOLT_REPO_URL}...")
    try:
        subprocess.run(["git", "clone", METRICBOLT_REPO_URL, REPO_DIR], check=True)
    except Exception as e:
        print(f"[!] Git clone failed: {e}. Attempting to continue if files already exist...")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

try:
    import pytorch_metric_learning
    import einops
except ImportError:
    print("[*] Installing transformer dependencies (pytorch-metric-learning, einops)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pytorch-metric-learning", "einops", "-q"], check=True)

# Required imports from cloned repo
try:
    from utils import Option
    from Models.BolT.hyperparams import getHyper_bolT
    from Models.BolT.bolT import BolT
    from pytorch_metric_learning import losses, distances
    print("[OK] All MetricBolT dependencies loaded.")
except ImportError as e:
    print(f"[!] Failed to import MetricBolT modules: {e}")
    print("Ensure you have internet access to clone the repo and install dependencies.")

# %% [markdown]
# ## 2. Data Discovery & Loading

# %%
# ==========================================
# PATH DISCOVERY (Standardized for Kaggle)
# ==========================================
RAW_REST_DIR = None
RAW_TASK_DIR = None

def setup_environment():
    """Discover HCP data paths on Kaggle or local filesystem."""
    global RAW_REST_DIR, RAW_TASK_DIR
    
    # Priority search paths
    search_paths = [
        "/kaggle/input",
        "/kaggle/working",
        "."
    ]
    
    # Locate Raw Data (subjects folders)
    print(">>> Searching for raw BOLD timeseries...")
    for base_path in search_paths:
        if not os.path.exists(base_path): continue
        for root, dirs, files in os.walk(base_path):
            if "subjects" in dirs:
                is_rest = "rest" in root.lower() or "hcp_rest" in root.lower()
                is_task = "task" in root.lower() or "motor" in root.lower() or "hcp_task" in root.lower()
                
                if is_rest and RAW_REST_DIR is None:
                    RAW_REST_DIR = root
                elif is_task and RAW_TASK_DIR is None:
                    RAW_TASK_DIR = root
        if RAW_REST_DIR and RAW_TASK_DIR: break

    # Check for DATA.zip if subjects not found
    if RAW_REST_DIR is None:
        import zipfile
        for root, _, files in os.walk(INPUT_DIR):
            if "DATA.zip" in files:
                zip_path = os.path.join(root, "DATA.zip")
                print(f">>> Extracting {zip_path} to {WORKING_DIR}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(WORKING_DIR)
                RAW_REST_DIR = os.path.join(WORKING_DIR, "rest")
                RAW_TASK_DIR = os.path.join(WORKING_DIR, "task")
                break

setup_environment()

def find_fc_file(directory, task_name):
    """Find FC file for a given task name with flexible naming conventions."""
    if not directory or not os.path.exists(directory):
        return None
    
    # Common naming patterns
    task_patterns = [
        f"fc_{task_name}.npy",
        f"{task_name}_fc.npy",
        f"{task_name}.npy"
    ]
    
    # Also handle 'rest' specially for common variations
    if task_name.lower() == 'rest':
        task_patterns.extend(["rest_fc.npy", "fc_rest.npy", "REST.npy"])
        
    for f in os.listdir(directory):
        if any(p.lower() == f.lower() for p in task_patterns):
            return os.path.join(directory, f)
    return None

print(f"Raw Rest Dir: {RAW_REST_DIR or 'Not Found'}")
print(f"Raw Task Dir: {RAW_TASK_DIR or 'Not Found'}")

# ==========================================
# BOLD SCAN DEFINITIONS
# ==========================================
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]

def get_image_ids(name):
    """Get BOLD run IDs for a given task name."""
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids: 
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_concat_timeseries(subject_id, run_idx_list, base_dir, dynamic_length=None):
    """Load and concatenate timeseries for a subject, with mean-centering and optional cropping."""
    ts_list = []
    for run_id in run_idx_list:
        bold_path = os.path.join(base_dir, "subjects", str(subject_id), "timeseries")
        bold_file = f"bold{run_id}_Atlas_MSMAll_Glasser360Cortical.npy"
        full_path = os.path.join(bold_path, bold_file)
        if os.path.exists(full_path):
            ts = np.load(full_path)
            ts = ts - np.mean(ts, axis=1, keepdims=True)
            ts_list.append(ts)
            
    if not ts_list:
        return None
        
    ts_concat = np.concatenate(ts_list, axis=1)
    
    if dynamic_length:
        if ts_concat.shape[1] > dynamic_length:
            ts_concat = ts_concat[:, :dynamic_length]
        elif ts_concat.shape[1] < dynamic_length:
            # Pad with zeros to reach target length
            pad = np.zeros((ts_concat.shape[0], dynamic_length - ts_concat.shape[1]))
            ts_concat = np.concatenate([ts_concat, pad], axis=1)
        
    return ts_concat

# %% [markdown]
# ## 3. Evaluation Metrics (Identical to Main Pipeline)

# %%
def calculate_accuracy(corr_matrix):
    """Top-1 identification accuracy and prediction array."""
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    n = corr_matrix.shape[0]
    preds = np.array([np.argmax(corr_matrix[i, :]) == i for i in range(n)])
    return np.mean(preds), preds

def calculate_top_k_accuracy(corr_matrix, k=5):
    """Top-k identification accuracy."""
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if i in np.argsort(corr_matrix[i, :])[-k:])
    return correct / n

def calculate_mean_rank(corr_matrix):
    """Mean rank of correct identity."""
    n = corr_matrix.shape[0]
    ranks = []
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
    return np.mean(ranks)

def calculate_mrr(corr_matrix):
    """Mean Reciprocal Rank."""
    n = corr_matrix.shape[0]
    mrr = 0
    for i in range(n):
        sorted_idx = np.argsort(corr_matrix[i, :])[::-1]
        rank = np.where(sorted_idx == i)[0][0] + 1
        mrr += 1.0 / rank
    return mrr / n

def differential_identifiability(corr_matrix):
    """Differential identifiability: mean(self) - mean(other)."""
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    return np.mean(self_corr) - np.mean(other_corr)

def compute_all_metrics(corr_matrix):
    """Compute comprehensive evaluation metrics."""
    acc, preds = calculate_accuracy(corr_matrix)
    return {
        'top_1_accuracy': acc,
        'preds': preds.tolist(),
        'top_3_accuracy': calculate_top_k_accuracy(corr_matrix, k=3),
        'top_5_accuracy': calculate_top_k_accuracy(corr_matrix, k=5),
        'top_10_accuracy': calculate_top_k_accuracy(corr_matrix, k=10),
        'mean_rank': calculate_mean_rank(corr_matrix),
        'mrr': calculate_mrr(corr_matrix),
        'differential_id': differential_identifiability(corr_matrix)
    }

def plot_full_correlation_matrix(corr_matrix, save_path, title='Subject Identification Heatmap'):
    """Plots a single identification heatmap (1-to-1 parity with ConvAE+SDL)."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='viridis')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Reference Scan (Resting State)', fontsize=12)
    plt.ylabel('Target Scan (Task)', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_similarity_distributions(corr_matrix, save_path, title='Similarity Distributions'):
    """Plot self vs other correlation distributions (1-to-1 parity with ConvAE+SDL)."""
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(other_corr, bins=50, alpha=0.5, label='Other-Identity', density=True, color='gray')
    plt.hist(self_corr, bins=20, alpha=0.7, label='Self-Identity', density=True, color='red')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Pearson Correlation (Embedding Space)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ablation_results(raw_acc, model_acc, save_path, task_name):
    """Plot ablation study results (1-to-1 parity with ConvAE+SDL)."""
    plt.figure(figsize=(10, 6))
    methods = ['Raw FC (Baseline)', 'MetricBolT (Learned)']
    accuracies = [raw_acc, model_acc]
    colors = ['#95a5a6', '#2E86C1']
    
    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
    plt.ylim(0, 1.05)
    plt.ylabel('Top-1 Accuracy', fontsize=12)
    plt.title(f'Ablation Study: Metric Learning Contribution ({task_name.upper()})', fontsize=14)
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# %% [markdown]
# ## 4. Robustness Analysis Functions

# %%
def get_batched_embeddings(tensor_data, model, device, batch_size=16):
    """Safely compute embeddings in batches to prevent OOM."""
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, tensor_data.shape[0], batch_size):
            batch = tensor_data[i:i+batch_size].to(device)
            emb = model(batch)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)

def noise_robustness(rest_tensor, task_tensor, model, device, noise_levels=[0, 0.05, 0.1, 0.2, 0.3], n_repeats=3):
    """Evaluates how identification accuracy degrades with timeseries noise."""
    num_subjects = rest_tensor.shape[0]
    results = {}
    model.eval()
    
    # Pre-calculate rest embeddings (no noise on reference set for standard evaluation)
    with torch.no_grad():
        rest_embs = get_batched_embeddings(rest_tensor, model, device)
        rest_embs = torch.nn.functional.normalize(rest_embs, p=2, dim=1).numpy()
        
    for noise in noise_levels:
        accs = []
        for _ in range(n_repeats):
            # Add noise to TASK timeseries
            noisy_task = task_tensor + torch.randn_like(task_tensor) * noise
            
            with torch.no_grad():
                task_embs = get_batched_embeddings(noisy_task, model, device)
                task_embs = torch.nn.functional.normalize(task_embs, p=2, dim=1).numpy()
                
            corr = np.corrcoef(task_embs, rest_embs)[:num_subjects, num_subjects:]
            acc, _ = calculate_accuracy(corr)
            accs.append(acc)
        
        results[noise] = {'mean': np.mean(accs), 'std': np.std(accs)}
    return results

def bootstrap_ci(corr_matrix, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for accuracy from a correlation matrix."""
    n = corr_matrix.shape[0]
    accuracies = []
    for _ in range(n_bootstrap):
        # Sample subjects with replacement
        idx = np.random.choice(n, n, replace=True)
        # In fingerprinting, we sample the subjects (rows of the corr matrix)
        # but the identity mapping must remain (diag elements)
        sub_corr = corr_matrix[idx, :][:, idx]
        acc, _ = calculate_accuracy(sub_corr)
        accuracies.append(acc)
    
    alpha = 1 - confidence
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'ci_lower': np.percentile(accuracies, 100 * alpha / 2),
        'ci_upper': np.percentile(accuracies, 100 * (1 - alpha / 2))
    }

def sample_size_robustness(rest_embs, task_embs, sample_sizes=[67, 135, 203, 271, 339], n_repeats=5):
    """Evaluates how identification accuracy scales with number of subjects."""
    num_subjects_total = rest_embs.shape[0]
    results = {}
    
    for N in sample_sizes:
        if N > num_subjects_total: continue
        accs = []
        for _ in range(n_repeats):
            idx = np.random.choice(num_subjects_total, N, replace=False)
            task_sample = task_embs[idx]
            rest_sample = rest_embs[idx]
            corr = np.corrcoef(task_sample, rest_sample)[:N, N:]
            acc, _ = calculate_accuracy(corr)
            accs.append(acc)
        results[N] = {'mean': np.mean(accs), 'std': np.std(accs)}
    return results

def plot_robustness_comparison(robustness_dict, save_path):
    """Plots robustness comparison for baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for method, (noise_res, sample_res) in robustness_dict.items():
        if noise_res:
            n_levels = sorted(list(noise_res.keys()))
            axes[0].errorbar(n_levels, [noise_res[nl]['mean'] for nl in n_levels], 
                           yerr=[noise_res[nl]['std'] for nl in n_levels], label=method, marker='o')
        if sample_res:
            s_sizes = sorted(list(sample_res.keys()))
            axes[1].errorbar(s_sizes, [sample_res[ss]['mean'] for ss in s_sizes], 
                           yerr=[sample_res[ss]['std'] for ss in s_sizes], label=method, marker='s')
    axes[0].set_title('Noise Robustness (MetricBolT)'); axes[0].set_xlabel('Sigma'); axes[0].set_ylabel('Accuracy')
    axes[1].set_title('Sample Robustness (MetricBolT)'); axes[1].set_xlabel('Subjects N'); axes[1].set_ylabel('Accuracy')
    axes[0].legend(); axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path); plt.close()

# %% [markdown]
# ## 5. MetricBolT Training & Evaluation (5-Fold CV)

# %%
def run_comparison(task_name="motor", num_subjects=100, dynamic_length=284, epochs=30, device=DEVICE):
    """
    Run MetricBolT baseline comparison with 5-Fold Cross-Validation.
    """
    print("=" * 60)
    # Ensure environment is ready
    setup_environment()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f">>> Starting run: {timestamp} for task: {task_name}")
    max_subjects = num_subjects
    total_training_time = 0.0
    total_inference_time = 0.0
    actual_num_subjects = 0

    rest_run_ids = get_image_ids("rest")
    task_run_ids = get_image_ids(task_name)
    
    # Data Loading Logic
    if not (RAW_REST_DIR and RAW_TASK_DIR and os.path.exists(RAW_REST_DIR) and os.path.exists(RAW_TASK_DIR)):
        raise FileNotFoundError("Could not locate HCP raw BOLD data directories.")

    subs_rest = set([d for d in os.listdir(os.path.join(RAW_REST_DIR, "subjects")) if d.isdigit()])
    subs_task = set([d for d in os.listdir(os.path.join(RAW_TASK_DIR, "subjects")) if d.isdigit()])
    intersect_subs = sorted(list(subs_rest.intersection(subs_task)))
    subjects = intersect_subs[:max_subjects]
    
    if len(subjects) < 10:
        raise ValueError(f"Insufficient subjects found (found {len(subjects)}, need at least 10 for 5-Fold CV).")

    rest_data, rest2_data, task_data, valid_subjects = [], [], [], []
    r_lr1, r_lr2 = rest_run_ids[0], rest_run_ids[2] if len(rest_run_ids)>2 else rest_run_ids[0]
    t_lr = task_run_ids[1] if len(task_run_ids)>1 else task_run_ids[0]

    for s in tqdm(subjects, desc=f"Loading {task_name.upper()}"):
        ts1 = load_concat_timeseries(s, [r_lr1], RAW_REST_DIR, dynamic_length)
        ts2 = load_concat_timeseries(s, [r_lr2], RAW_REST_DIR, dynamic_length)
        tst = load_concat_timeseries(s, [t_lr], RAW_TASK_DIR, dynamic_length)
        if ts1 is not None and ts2 is not None and tst is not None:
            # All must match dynamic_length (guaranteed by load_concat_timeseries padding/cropping)
            rest_data.append(ts1)
            rest2_data.append(ts2)
            task_data.append(tst)
            valid_subjects.append(s)
    
    if len(valid_subjects) < 10:
        raise ValueError(f"Could not load enough valid scans for {task_name.upper()}.")

    actual_num_subjects = len(valid_subjects)
    rest_tensor = torch.tensor(np.array(rest_data), dtype=torch.float32)
    rest_tensor2 = torch.tensor(np.array(rest2_data), dtype=torch.float32)
    task_tensor = torch.tensor(np.array(task_data), dtype=torch.float32)

    # 5-Fold CV
    n_folds = 5
    indices = np.random.RandomState(42).permutation(actual_num_subjects)
    fold_size = actual_num_subjects // n_folds
    final_rest_embs = np.zeros((actual_num_subjects, 360))
    final_task_embs = np.zeros((actual_num_subjects, 360))
    fold_losses = []
    last_model = None

    for fold in range(n_folds):
        test_idx = indices[fold*fold_size : (fold+1)*fold_size if fold<n_folds-1 else actual_num_subjects]
        train_idx = np.setdiff1d(indices, test_idx)
        
        # Model setup
        hyper = getHyper_bolT()
        hyper.dim = 360
        hyper.loss = "NTXentLoss"
        hyper.pooling = "cls"
        hyper.method = "BolT"
        
        details = Option({'device': device, 'nOfTrains': len(train_idx), 'batchSize': 16, 'nOfEpochs': epochs, 'method': 'BolT', 'nOfClasses': 0})
        model = BolT(hyper, details).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4) # Increased LR
        crit = losses.NTXentLoss(temperature=0.1) # Changed to Contrastive Loss
        
        # Include Task data in training! Otherwise it never sees the task domain
        train_x = torch.cat([rest_tensor[train_idx], rest_tensor2[train_idx], task_tensor[train_idx]], dim=0)
        train_y = torch.cat([torch.arange(len(train_idx)), torch.arange(len(train_idx)), torch.arange(len(train_idx))], dim=0)
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        
        try:
            from pytorch_metric_learning.samplers import MPerClassSampler
            sampler = MPerClassSampler(train_y, m=3, batch_size=18, length_before_new_iter=len(dataset)) # m=3 because we have 3 scans per subject
            dl = DataLoader(dataset, batch_size=18, sampler=sampler)
        except ImportError:
            dl = DataLoader(dataset, batch_size=16, shuffle=True)
        
        start_train_fold = time.time()
        model.train()
        for ep in range(epochs):
            ep_l = 0
            for bx, by in dl:
                opt.zero_grad()
                emb = torch.nn.functional.normalize(model(bx.to(device)), p=2, dim=1)
                loss = crit(emb, by.to(device))
                loss.backward(); opt.step()
                ep_l += loss.item()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_training_time += time.time() - start_train_fold
        fold_losses.append(ep_l/max(1, len(dl)))
        
        model.eval()
        start_inf = time.time()
        with torch.no_grad():
            r_fold_embs = get_batched_embeddings(rest_tensor[test_idx], model, device)
            final_rest_embs[test_idx] = torch.nn.functional.normalize(r_fold_embs, p=2, dim=1).numpy()
            
            t_fold_embs = get_batched_embeddings(task_tensor[test_idx], model, device)
            final_task_embs[test_idx] = torch.nn.functional.normalize(t_fold_embs, p=2, dim=1).numpy()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        total_inference_time += time.time() - start_inf
        last_model = model
        
        # Cleanup fold-specific memory
        del opt, crit, dl, train_x, train_y, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Metrics
    id_mat = np.corrcoef(final_task_embs, final_rest_embs)[:actual_num_subjects, actual_num_subjects:]
    metrics = compute_all_metrics(id_mat)
    
    # Raw FC Baseline
    tril = np.tril_indices(360, k=-1)
    raw_r = np.array([np.corrcoef(rest_tensor[i].cpu().numpy())[tril] if rest_tensor[i].shape[0] == 360 else np.corrcoef(np.random.randn(360,100))[tril] for i in range(actual_num_subjects)])
    raw_t = np.array([np.corrcoef(task_tensor[i].cpu().numpy())[tril] if task_tensor[i].shape[0] == 360 else np.corrcoef(np.random.randn(360,100))[tril] for i in range(actual_num_subjects)])
    raw_mat = np.corrcoef(raw_t, raw_r)[:actual_num_subjects, actual_num_subjects:]
    raw_met = compute_all_metrics(raw_mat)

    # Robustness Analysis
    n_rob = noise_robustness(rest_tensor, task_tensor, last_model, device)
    s_rob = sample_size_robustness(final_rest_embs, final_task_embs)
    
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}_{task_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Visualizations (1-to-1 parity with ConvAE+SDL)
    print(f"[*] Generating visualizations in {run_dir}...")
    plot_full_correlation_matrix(id_mat, os.path.join(run_dir, "heatmap_metricbolt.png"), title=f"MetricBolT Heatmap ({task_name.upper()})")
    plot_full_correlation_matrix(raw_mat, os.path.join(run_dir, "heatmap_raw_fc.png"), title=f"Raw FC Heatmap ({task_name.upper()})")
    plot_similarity_distributions(id_mat, os.path.join(run_dir, "similarity_distributions.png"), title=f"MetricBolT Embeddings: {task_name.upper()}")
    plot_ablation_results(raw_met['top_1_accuracy'], metrics['top_1_accuracy'], os.path.join(run_dir, "ablation_study.png"), task_name)
    
    # Combined Robustness Plot (matching style)
    plot_robustness_comparison({'MetricBolT': (n_rob, s_rob)}, os.path.join(run_dir, "robustness_comparison.png"))
    
    json_results = {
        'task': task_name,
        'num_subjects': actual_num_subjects,
        'total_train_time': total_training_time,
        'avg_inference_time_ms': (total_inference_time / actual_num_subjects) * 1000 if actual_num_subjects > 0 else 0,
        'metricbolt_metrics': metrics,
        'raw_fc_metrics': raw_met,
        'noise_robustness': n_rob,
        'sample_robustness': s_rob,
        'timestamp': timestamp
    }
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"[OK] Results saved to {run_dir}")
    return metrics

# %% [markdown]
# ## 6. Execute Pipeline

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MetricBolT Baseline")
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--num_subjects", type=int, default=339)
    parser.add_argument("--epochs", type=int, default=30)
    args, _ = parser.parse_known_args()
    
    task_list = ["social", "emotion", "gambling", "relational"] if args.task == "all" else [args.task]
    
    # "wm", "language", "motor"
    
    # Native lengths for cropping/padding
    TASK_LENGTHS = {
        "rest": 1200, "motor": 284, "wm": 405, "emotion": 176,
        "gambling": 253, "language": 316, "relational": 232, "social": 274
    }

    for t in task_list:
        try:
            run_comparison(
                task_name=t, 
                num_subjects=args.num_subjects, 
                epochs=args.epochs,
                dynamic_length=TASK_LENGTHS.get(t, 284)
            )
        except Exception as e:
            print(f"[ERROR] Task {t} failed: {e}")

    # --- Benchmark ---
    print("\n" + "="*60)
    print("BENCHMARKING COMPUTATIONAL COMPLEXITY")
    print("="*60)
    from torch import randn
    try:
        hyper = getHyper_bolT()
        hyper.dim = 360
        hyper.pooling = "cls"
        hyper.method = "BolT"
        
        details = Option({})
        details.device = DEVICE
        details.nOfTrains = 100
        details.batchSize = 1
        details.nOfEpochs = 1
        details.nOfClasses = 0
        details.method = "BolT"
        
        bolt = BolT(hyper, details).to(DEVICE)
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = bolt(randn(1, 360, 284).to(DEVICE))
        avg_time = (time.time() - start) / 10 * 1000
        params = sum(p.numel() for p in bolt.parameters() if p.requires_grad)
        print(f"MetricBolT (Transformer):")
        print(f"  Parameters: {params:,}")
        print(f"  Inference:  {avg_time:.2f} ms\n")
        del bolt
        gc.collect()
    except Exception as e:
        print(f"Benchmark failed: {e}")
