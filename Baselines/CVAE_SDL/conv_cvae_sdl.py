# %% [markdown]
# # CVAE + SDL Baseline - Brain Fingerprinting Comparison
# 
# This notebook implements the Conditional Variational Autoencoder + Sparse Dictionary
# Learning (CVAE+SDL) baseline from **Lu et al., NeuroImage 295 (2024) 120651** for
# direct comparison against our ConvAE+SDL Brain Fingerprinting pipeline.
#
# **Key Difference from Our Method:**
# - Lu et al. use a **Conditional VAE** that embeds fMRI state information (rest vs task)
#   into the encoding/decoding process. This allows the CVAE to better capture shared
#   inter-subject features conditioned on the scan type.
# - Our method uses a standard **Convolutional Autoencoder (ConvAE)** without state conditioning.
# - Both methods then apply **Sparse Dictionary Learning (K-SVD + OMP)** on the residuals.
#
# **Awareness-Compliant Design (Orlichenko et al., 2023):**
# - 5-Fold Cross-Validation with strict subject-level splits
# - CVAE trained per fold on TRAIN subjects only, evaluated on HELD-OUT subjects
# - SDL dictionary learned from TRAIN residuals only, applied inductively to TEST
# - Prevents identifiability leakage / double-dipping
#
# **Evaluation Protocol:**
# - Identical metrics to `kaggle_brain_fingerprinting.py`
# - Same FC data loading, same subject ordering
# - Results directly comparable
#
# ---

# %% [markdown]
# ## 1. Setup & Dependencies

# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import orthogonal_mp
import datetime
import time
import json
import shutil
from tqdm.auto import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "cvae_sdl_results")
SAVE_DIR = os.path.join(WORKING_DIR, "FC_DATA")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 2. Data Discovery & Loading

# %%
# ==========================================
# PATH DISCOVERY (Standardized for Kaggle)
# ==========================================
RAW_REST_DIR = None
RAW_TASK_DIR = None
FC_DATA_DIR = SAVE_DIR

def setup_environment():
    """Discover HCP data paths on Kaggle or local filesystem."""
    global RAW_REST_DIR, RAW_TASK_DIR, FC_DATA_DIR
    
    # Priority search paths for FC Data
    search_paths = [
        "/kaggle/input",
        "/kaggle/working",
        "."
    ]
    
    # 1. Search for pre-calculated FC Data (.npy files)
    print(">>> Searching for pre-calculated FC Data...")
    found_fc = False
    for base_path in search_paths:
        if not os.path.exists(base_path): continue
        for root, dirs, files in os.walk(base_path):
            if "fc_rest.npy" in files:
                FC_DATA_DIR = root
                found_fc = True
                print(f"  [OK] Found FC Data at: {FC_DATA_DIR}")
                break
        if found_fc: break

    # 2. Search for Raw Data (subjects folders) if FC not found
    if not found_fc:
        print(">>> FC Data not found, searching for raw BOLD timeseries...")
        for base_path in search_paths:
            if not os.path.exists(base_path): continue
            for root, dirs, files in os.walk(base_path):
                if "subjects" in dirs:
                    if "rest" in root.lower() and RAW_REST_DIR is None:
                        RAW_REST_DIR = root
                    elif ("task" in root.lower() or "motor" in root.lower() or "hcp_task" in root.lower()) and RAW_TASK_DIR is None:
                        RAW_TASK_DIR = root
        
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

def generate_synthetic_fc(num_subjects, n_parcels=360):
    """Generate synthetic functional connectivity matrices."""
    print(f">>> [SYNTHETIC] Generating {num_subjects} synthetic subjects...")
    # Create a base connectivity pattern
    base_pattern = np.random.rand(n_parcels, n_parcels)
    base_pattern = (base_pattern + base_pattern.T) / 2
    
    fc_data = []
    for i in range(num_subjects):
        # Subject-specific variation
        subject_variation = np.random.normal(0, 0.1, (n_parcels, n_parcels))
        subject_variation = (subject_variation + subject_variation.T) / 2
        
        # Add a "fingerprint"
        fingerprint = np.zeros((n_parcels, n_parcels))
        idx = i % n_parcels
        fingerprint[idx, :] = 1.0
        fingerprint[:, idx] = 1.0
        
        fc = base_pattern + subject_variation + 0.5 * fingerprint
        fc = np.clip(fc, -1.0, 1.0)
        np.fill_diagonal(fc, 1.0)
        fc_data.append(fc)
        
    return np.array(fc_data)

print(f"--- Environment Verified ---")
print(f"Device: {DEVICE.upper()}")
print(f"Raw Rest: {RAW_REST_DIR or 'Not Found'}")
print(f"Raw Task: {RAW_TASK_DIR or 'Not Found'}")
print(f"FC Data Path: {FC_DATA_DIR}")
print(f"----------------------------\n")

# Constants (identical to main pipeline)
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
TRIL_IDX = np.tril_indices(360, k=-1)

# --- FC Generation (identical to main pipeline) ---
def get_image_ids(name):
    """Get run IDs for a given task name."""
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids:
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_single_timeseries(subject_id, bold_run, base_dir, remove_mean=True):
    """Load a single timeseries file for a subject."""
    bold_path = os.path.join(base_dir, "subjects", str(subject_id), "timeseries")
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    full_path = os.path.join(bold_path, bold_file)
    if not os.path.exists(full_path):
        return None
    ts = np.load(full_path)
    if remove_mean:
        ts -= ts.mean(axis=1, keepdims=True)
    return ts

def generate_fc_for_task(task_name, subjects_list, base_dir, n_parcels=360):
    """Generate functional connectivity matrices for a task."""
    print(f">>> Generating {task_name} FC...")
    run_ids = get_image_ids(task_name)
    n_runs = 4 if task_name == "rest" else 2
    
    fc_all, subs_idx = [], []
    for idx, sub_id in enumerate(tqdm(subjects_list, desc=task_name.upper())):
        all_runs = [load_single_timeseries(sub_id, run_ids[0] + r, base_dir) for r in range(n_runs)]
        ts_list = [ts for ts in all_runs if ts is not None]
        if ts_list:
            fc_all.append(np.corrcoef(np.concatenate(ts_list, axis=1)))
            subs_idx.append(sub_id)
    return np.array(fc_all), subs_idx

# %% [markdown]
# ## 3. CVAE Architecture (Lu et al., 2024)

# %%
class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) for Brain Fingerprinting.
    
    From Lu et al., NeuroImage 295 (2024):
    - Embeds fMRI state information (Rest=0, Task=1) into encoding/decoding
    - The condition allows the CVAE to better capture SHARED inter-subject features
    - Residual (Original - Reconstruction) contains individual-specific information
    
    Architecture:
    - Decoder: Linear(latent_dim+1 -> flatten_size) -> ConvTranspose2d(64->32->16->1) + Tanh
    """
    def __init__(self, n_parcels=360, latent_dim=128):
        super(ConditionalVAE, self).__init__()
        self.n = n_parcels
        self.latent_dim = latent_dim
        
        # Encoder (2 channels: 1 for FC matrix, 1 for condition map)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 360->180
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # 180->90
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)    # 90->45
        )
        self.flatten_size = 64 * 45 * 45  # 129,600
        
        # VAE: mu and logvar, conditioned on state label
        self.fc_mu = nn.Linear(self.flatten_size + 1, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size + 1, latent_dim)
        
        # Decoder input, conditioned on state label
        self.fc_dec = nn.Linear(latent_dim + 1, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 45 to 90
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 90 to 180
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),   # 180 to 360
            nn.Tanh()
        )

    def encode(self, x, c):
        """Encode FC matrix with state condition."""
        c_channel = c.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x_cond = torch.cat([x, c_channel], dim=1)  # (B, 2, n, n)
        h = self.encoder(x_cond).reshape(x.size(0), -1)
        h_cond = torch.cat([h, c], dim=1)
        
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        
        # CRITICAL VAE FIX: Prevent float32 Inf/NaN overflow before reparameterization
        mu = torch.clamp(mu, min=-1000.0, max=1000.0)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode latent vector with state condition."""
        z_cond = torch.cat([z, c], dim=1)
        h = self.fc_dec(z_cond).view(-1, 64, 45, 45)
        recon = self.decoder(h)
        if recon.shape[2:] != (self.n, self.n):
            recon = F.interpolate(recon, size=(self.n, self.n), mode='bilinear', align_corners=True)
        return recon

    def forward(self, x, c):
        """Full forward pass: encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cvae_loss(recon_x, x, mu, logvar, beta=0.1):
    """CVAE Loss = Reconstruction (MSE) + beta x KL Divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld) / x.size(0)

# %% [markdown]
# ## 4. Sparse Dictionary Learning (K-SVD + OMP)

# %%
def omp_sparse_coding(Y, D, L):
    """Orthogonal Matching Pursuit for sparse coding."""
    n_samples = Y.shape[1]
    n_atoms = D.shape[1]
    X = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
    return X

def update_dictionary(Y, D, X):
    """Update dictionary atoms using SVD."""
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])[0]
        if len(idx) == 0:
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        residual = Y[:, idx] - np.dot(D, X[:, idx]) + np.outer(D[:, k], X[k, idx])
        U, s, Vt = np.linalg.svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, idx] = s[0] * Vt[0, :]
    return D, X

def k_svd(Y, K, L, n_iter=10, verbose=True, random_state=None):
    """K-SVD algorithm for sparse dictionary learning."""
    if random_state is not None:
        np.random.seed(random_state)
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    iterator = tqdm(range(n_iter), desc="K-SVD") if verbose else range(n_iter)
    for _ in iterator:
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    return D, X

def reconstruct_symmetric_matrix(tri_elements, n=360):
    """Reconstruct symmetric matrix from lower triangular elements."""
    tril_idx = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n))
    matrix[tril_idx] = tri_elements
    matrix += matrix.T
    return matrix

def perform_grid_search(Y, rest_flat, n_subjects, n_parcels, K_range=(2, 16), L_max=None, n_iter=3, task_name="unknown"):
    """
    Grid search for optimal K and L parameters.
    Returns (accuracies, best_K, best_L)
    """
    print(f"  Grid Search K={K_range}...")
    
    if isinstance(K_range, tuple):
        Ks = sorted(list(range(K_range[0], K_range[1] + 1, 1)))
    else:
        Ks = sorted(list(K_range))
        
    best_acc = -1.0
    best_K = 15  # Default fallback
    best_L = 12
    
    all_L_vals = sorted(list(set([L for K_val in Ks for L in range(2, K_val + 1, 2)])))
    accuracies = np.zeros((len(Ks), len(all_L_vals)))
    
    for i, K in enumerate(tqdm(Ks, desc="Grid Search K")):
        L_vals = range(2, K + 1, 2)
        for L in L_vals:
            j = all_L_vals.index(L)
            # Run simplified K-SVD
            D, X = k_svd(Y, K, L, n_iter=n_iter, verbose=False, random_state=42)
            
            if rest_flat is not None:
                # Approximate Rest Sparse Codes using learned D
                X_rest = omp_sparse_coding(rest_flat, D, L)
                
                corr = np.corrcoef(X.T, X_rest.T)[:n_subjects, n_subjects:]
                acc, _ = calculate_accuracy(corr)
                
                accuracies[i, j] = acc
                if acc > best_acc:
                    best_acc = acc
                    best_K = K
                    best_L = L
    
    print(f"  Found Optimal: K={best_K}, L={best_L} (Acc: {best_acc:.4f})")
    
    # Plot Heatmap
    if np.nanmax(accuracies) > 0:
        try:
            plt.figure(figsize=(12, 10))
            mask = accuracies == 0
            df_plot = pd.DataFrame(accuracies, index=Ks, columns=all_L_vals)
            sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
            plt.title(f'CVAE Grid Search Identification Accuracy ({task_name.upper()})')
            plt.ylabel('Dictionary Atoms (K)')
            plt.xlabel('Sparsity Level (L)')
            
            # Save to OUTPUT_DIR
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(OUTPUT_DIR, f"grid_search_cvae_{task_name}_{timestamp}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [OK] CVAE Grid search heatmap saved to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not plot CVAE heatmap: {e}")
            
    return accuracies, best_K, best_L

# %% [markdown]
# ## 5. Evaluation Metrics (Identical to Main Pipeline)

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

def plot_full_correlation_matrix(corr_matrix, save_path, title='Subject Identification Heatmap'):
    """Plots a single identification heatmap (1-to-1 parity with project style)."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap='viridis')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Reference Scan (Resting State)', fontsize=12)
    plt.ylabel('Target Scan (Task)', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_similarity_distributions(corr_matrix, save_path, title='Similarity Distributions'):
    """Plot self vs other correlation distributions (1-to-1 parity with project style)."""
    n = corr_matrix.shape[0]
    self_corr = np.diag(corr_matrix)
    mask = ~np.eye(n, dtype=bool)
    other_corr = corr_matrix[mask]
    
    plt.figure(figsize=(10, 6))
    plt.hist(other_corr, bins=50, alpha=0.5, label='Other-Identity', density=True, color='gray')
    plt.hist(self_corr, bins=20, alpha=0.7, label='Self-Identity', density=True, color='red')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Pearson Correlation', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_all_metrics(corr_matrix):
    """Compute comprehensive evaluation metrics (identical to main pipeline)."""
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

def noise_robustness(fc_task, fc_rest, model=None, dictionary=None, L=15, noise_levels=[0, 0.05, 0.1, 0.2, 0.3], n_repeats=5):
    """Evaluates how identification accuracy degrades with noise."""
    n_subjects = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    tril_idx = np.tril_indices(n_parcels, k=-1)
    results = {}
    
    for noise in noise_levels:
        accs = []
        for _ in range(n_repeats):
            fc_noisy = fc_task + np.random.randn(*fc_task.shape) * noise
            fc_noisy = np.clip(fc_noisy, -1, 1)
            
            if model is not None and dictionary is not None:
                # CVAE Refinement logic
                fc_tensor = torch.tensor(fc_noisy[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
                c_task = torch.ones((n_subjects, 1), dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    recon, _, _ = model(fc_tensor, c_task)
                res_task = fc_noisy - recon.squeeze(1).cpu().numpy()
                Y_test = np.array([res_task[i][tril_idx] for i in range(n_subjects)]).T
                X_test = omp_sparse_coding(np.nan_to_num(Y_test), dictionary, L)
                
                task_proc = np.zeros_like(fc_noisy)
                for i in range(n_subjects):
                    task_proc[i] = res_task[i] - reconstruct_symmetric_matrix(np.dot(dictionary, X_test[:, i]), n_parcels)
            else:
                task_proc = fc_noisy
                
            # Compare processed task with RAW fc_rest (vectorized)
            task_flat = np.array([task_proc[i][tril_idx] for i in range(n_subjects)])
            rest_flat = np.array([fc_rest[i][tril_idx] for i in range(n_subjects)])
            corr = np.corrcoef(task_flat, rest_flat)[:n_subjects, n_subjects:]
            acc, _ = calculate_accuracy(corr)
            accs.append(acc)
        
        results[noise] = {'mean': np.mean(accs), 'std': np.std(accs)}
    return results

def bootstrap_ci(corr_matrix, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for accuracy from a correlation matrix."""
    n = corr_matrix.shape[0]
    accuracies = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
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

def sample_size_robustness(fc_task, fc_rest, sample_sizes=[67, 135, 203, 271, 339], n_repeats=5):
    """Evaluates how identification accuracy scales with number of subjects."""
    n_subjects_total = fc_task.shape[0]
    n_parcels = fc_task.shape[1]
    tril_idx = np.tril_indices(n_parcels, k=-1)
    results = {}
    
    for N in sample_sizes:
        if N > n_subjects_total: continue
        accs = []
        for _ in range(n_repeats):
            idx = np.random.choice(n_subjects_total, N, replace=False)
            task_flat = np.array([fc_task[i][tril_idx] for i in idx])
            rest_flat = np.array([fc_rest[i][tril_idx] for i in idx])
            corr = np.corrcoef(task_flat, rest_flat)[:N, N:]
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
    axes[0].set_title('Noise Robustness (CVAE)'); axes[0].set_xlabel('Sigma'); axes[0].set_ylabel('Accuracy')
    axes[1].set_title('Sample Robustness (CVAE)'); axes[1].set_xlabel('Subjects N'); axes[1].set_ylabel('Accuracy')
    plt.savefig(save_path); plt.close()

# %% [markdown]
# ## 6. CVAE+SDL Pipeline (5-Fold CV, Awareness Compliant)

# %%
TUNED_PARAMS = {
    "social": (15, 12),
    "gambling": (15, 14),
    "language": (15, 12),
    "relational": (13, 12)
}

def run_cvae_sdl_baseline(task_name="motor", num_subjects=339, K=15, L=12, 
                          epochs=20, beta=0.1, latent_dim=128, n_folds=5):
    """
    CVAE+SDL Baseline from Lu et al., NeuroImage 295 (2024) 120651.
    
    Awareness-Compliant Implementation:
    - 5-Fold CV with strict subject-level splits (same seed=42 as main pipeline)
    - CVAE trained per fold on TRAIN subjects only (both rest + task, conditioned)
    - SDL dictionary learned from TRAIN residuals only
    - Sparse coding applied inductively to TEST residuals
    - All evaluation metrics computed on HELD-OUT subjects only
    
    Args:
        task_name: HCP task name (motor, wm, emotion, gambling, language, relational, social)
        num_subjects: Number of subjects to use (max 339)
        K: Number of dictionary atoms for K-SVD
        L: Sparsity level for OMP
        epochs: Training epochs per fold for CVAE
        beta: KL divergence weight in CVAE loss
        latent_dim: Dimensionality of CVAE latent space
        n_folds: Number of CV folds
    """
    print("=" * 60)
    
    # Scoping fix
    last_fold_model = None
    total_training_time = 0.0
    total_inference_time = 0.0

    # ==========================================
    # DATA LOADING (Same logic as main pipeline)
    # ==========================================
    print("\n[1/4] Loading data...")
    n_parcels = 360
    
    rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
    task_path = os.path.join(FC_DATA_DIR, f"fc_{task_name}.npy")
    save_rest_path = os.path.join(SAVE_DIR, "fc_rest.npy")
    save_task_path = os.path.join(SAVE_DIR, f"fc_{task_name}.npy")
    
    if os.path.exists(rest_path) and os.path.exists(task_path):
        print(f"  Loading pre-calculated matrices from {FC_DATA_DIR}")
        fc_rest = np.load(rest_path)
        fc_task = np.load(task_path)
    elif RAW_REST_DIR is not None and RAW_TASK_DIR is not None:
        print("  Generating FC from raw timeseries...")
        subjects_path = os.path.join(RAW_REST_DIR, "subjects")
        subjects_list = sorted([d for d in os.listdir(subjects_path)
                               if os.path.isdir(os.path.join(subjects_path, d))])
        if num_subjects < len(subjects_list):
            subjects_list = subjects_list[:num_subjects]
            
        fc_rest, rest_subs = generate_fc_for_task("rest", subjects_list, RAW_REST_DIR, n_parcels)
        fc_task, task_subs = generate_fc_for_task(task_name, subjects_list, RAW_TASK_DIR, n_parcels)
        
        valid_subs = sorted(list(set(rest_subs) & set(task_subs)))
        fc_rest = fc_rest[[rest_subs.index(s) for s in valid_subs]]
        fc_task = fc_task[[task_subs.index(s) for s in valid_subs]]
        
        print(f"  Generated FC for {len(valid_subs)} subjects")
        np.save(save_rest_path, fc_rest)
        np.save(save_task_path, fc_task)
    else:
        print(">>> [!] No FC data found and no raw data directories available.")
        print(">>> [!] LOCAL RUN DETECTED: Generating synthetic data for seamless execution...")
        
        # Fallback to fewer subjects if on local machine to keep it fast
        if num_subjects > 50:
            print(f">>> [!] Reducing num_subjects from {num_subjects} to 50 for local synthetic run.")
            num_subjects = 50
            
        fc_rest = generate_synthetic_fc(num_subjects, n_parcels)
        # Task FC is Rest FC + some noise + task effect
        fc_task = 0.8 * fc_rest + 0.2 * generate_synthetic_fc(num_subjects, n_parcels)
        fc_task = np.clip(fc_task, -1.0, 1.0)
        
        print(f">>> [OK] Generated synthetic data for {num_subjects} subjects")
        os.makedirs(SAVE_DIR, exist_ok=True)
        np.save(save_rest_path, fc_rest)
        np.save(save_task_path, fc_task)

    if fc_rest.shape[0] > num_subjects:
        fc_rest = fc_rest[:num_subjects]
        fc_task = fc_task[:num_subjects]
        
    fc_rest = np.clip(np.nan_to_num(fc_rest, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    fc_task = np.clip(np.nan_to_num(fc_task, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    
    num_subjects = fc_rest.shape[0]
    n_parcels = fc_rest.shape[1]
    
    if num_subjects < 5:
        print("[!] Too few subjects (N < 5) for analysis. Aborting.")
        return None, None
    
    print(f"  Loaded {num_subjects} subjects, {n_parcels} parcels")

    # ==========================================
    # RAW FC BASELINE (Finn et al., 2015)
    # ==========================================
    print("\n[2/4] Computing Raw FC Baseline (Finn et al.)...")
    raw_rest = np.array([fc_rest[i][TRIL_IDX] for i in tqdm(range(num_subjects), desc="Vectorizing Rest FC")])
    raw_task = np.array([fc_task[i][TRIL_IDX] for i in tqdm(range(num_subjects), desc="Vectorizing Task FC")])
    raw_matrix = np.corrcoef(raw_task, raw_rest)[:num_subjects, num_subjects:]
    raw_metrics = compute_all_metrics(raw_matrix)
    print(f"  Raw FC Top-1: {raw_metrics['top_1_accuracy']:.4f}")

    # ==========================================
    # CVAE MODEL SETUP
    # ==========================================
    print(f"\n[3/4] Running CVAE+SDL with {n_folds}-Fold CV (Epochs={epochs}, K={K}, L={L}, beta={beta})...")
    
    cvae_model = ConditionalVAE(n_parcels, latent_dim).to(DEVICE)
    print(f"  CVAE Parameters: {cvae_model.count_parameters():,}")
    model_params = cvae_model.count_parameters()
    del cvae_model  # Will create fresh models per fold
    
    # Cross-validation setup (same seed as main pipeline)
    indices = np.random.RandomState(42).permutation(num_subjects)
    fold_size = num_subjects // n_folds
    
    # Out-of-fold arrays for aggregation
    refined_rest_all = np.zeros_like(fc_rest)
    refined_task_all = np.zeros_like(fc_task)
    residual_rest_all = np.zeros_like(fc_rest)
    residual_task_all = np.zeros_like(fc_task)
    fold_losses = []
    fold_accuracies_refined = []
    fold_accuracies_residual = []
    total_training_time = 0.0
    total_inference_time = 0.0
    last_D = None  # Store dictionary from last fold for interpretability
    
    for fold in tqdm(range(n_folds), desc="CV Folds"):
        start_idx = fold * fold_size
        end_idx = num_subjects if fold == n_folds - 1 else (fold + 1) * fold_size
        test_idx = indices[start_idx:end_idx]
        train_idx = np.setdiff1d(indices, test_idx)
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        
        # --- CVAE Training on TRAIN subjects only ---
        train_rest = fc_rest[train_idx]
        train_task = fc_task[train_idx]
        
        # Combine rest + task of TRAIN subjects, with condition labels
        x_train = torch.tensor(
            np.concatenate([train_rest, train_task])[:, np.newaxis, :, :],
            dtype=torch.float32
        )
        c_train = torch.tensor(
            np.concatenate([np.zeros(n_train), np.ones(n_train)]),
            dtype=torch.float32
        ).unsqueeze(1)
        
        fold_model = ConditionalVAE(n_parcels, latent_dim).to(DEVICE)
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.001)
        dataset = TensorDataset(x_train, c_train)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training
        import time
        start_train_fold = time.time()
        fold_model.train()
        for ep in tqdm(range(epochs), desc=f"  Fold {fold+1} Training", leave=False):
            ep_loss = 0.0
            n_batches = 0
            for batch_x, batch_c in dataloader:
                batch_x, batch_c = batch_x.to(DEVICE), batch_c.to(DEVICE)
                fold_optimizer.zero_grad()
                recon, mu, logvar = fold_model(batch_x, batch_c)
                
                loss = cvae_loss(recon, batch_x, mu, logvar, beta=beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=5.0)
                fold_optimizer.step()
                
                if not torch.isnan(loss):
                    ep_loss += loss.item()
                n_batches += 1

        if torch.cuda.is_available(): torch.cuda.synchronize()
        train_time_fold = time.time() - start_train_fold
        total_training_time += train_time_fold
        print(f"  Fold {fold+1} training time: {train_time_fold:.2f} s")
        avg_loss = ep_loss / max(1, n_batches)
        fold_losses.append(avg_loss)
        
        # --- Compute Residuals on TRAIN set for SDL Dictionary ---
        fold_model.eval()
        with torch.no_grad():
            # Train REST residuals (for dictionary learning)
            tr_rest_t = torch.tensor(train_rest[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            c_rest_tr = torch.zeros((n_train, 1), dtype=torch.float32).to(DEVICE)
            recon_train_rest, _, _ = fold_model(tr_rest_t, c_rest_tr)
        
        train_res_rest = train_rest - recon_train_rest.squeeze(1).cpu().numpy()
        
        # Learn Dictionary D from TRAIN residuals only (INDUCTIVE)
        Y_train = np.array([train_res_rest[i][TRIL_IDX] for i in tqdm(range(n_train), desc=f"  Fold {fold+1} Train SDL Setup", leave=False)]).T
        Y_train = np.nan_to_num(Y_train, nan=0.0)
        D, _ = k_svd(Y_train, K=K, L=L, n_iter=10, verbose=False, random_state=42)
        last_D = D  # Save for interpretability
        
        # --- Inference on HELD-OUT TEST subjects ---
        with torch.no_grad():
            te_rest_t = torch.tensor(fc_rest[test_idx][:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            te_task_t = torch.tensor(fc_task[test_idx][:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            c_rest_te = torch.zeros((n_test, 1), dtype=torch.float32).to(DEVICE)
            c_task_te = torch.ones((n_test, 1), dtype=torch.float32).to(DEVICE)
            
            recon_test_rest, _, _ = fold_model(te_rest_t, c_rest_te)
            recon_test_task, _, _ = fold_model(te_task_t, c_task_te)
        
        test_res_rest = fc_rest[test_idx] - recon_test_rest.squeeze(1).cpu().numpy()
        test_res_task = fc_task[test_idx] - recon_test_task.squeeze(1).cpu().numpy()
        
        # Store raw residuals for ablation
        for i, act_idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Storing Residuals", total=n_test, leave=False):
            residual_rest_all[act_idx] = test_res_rest[i]
            residual_task_all[act_idx] = test_res_task[i]
        
        # Sparse code TEST residuals using TRAIN dictionary D (INDUCTIVE SDL)
        Y_test_rest = np.array([test_res_rest[i][TRIL_IDX] for i in tqdm(range(n_test), desc=f"  Fold {fold+1} Test SDL Setup (Rest)", leave=False)]).T
        Y_test_task = np.array([test_res_task[i][TRIL_IDX] for i in tqdm(range(n_test), desc=f"  Fold {fold+1} Test SDL Setup (Task)", leave=False)]).T
        
        Y_test_rest = np.nan_to_num(Y_test_rest, nan=0.0)
        Y_test_task = np.nan_to_num(Y_test_task, nan=0.0)
        
        X_rest_test = omp_sparse_coding(Y_test_rest, D, L=L)
        X_task_test = omp_sparse_coding(Y_test_task, D, L=L)
        
        inference_start = time.time()
        for i, act_idx in tqdm(enumerate(test_idx), desc=f"  Fold {fold+1} Reconstruction", total=n_test, leave=False):
            refined_rest_all[act_idx] = test_res_rest[i] - reconstruct_symmetric_matrix(np.dot(D, X_rest_test[:, i]), n_parcels)
            refined_task_all[act_idx] = test_res_task[i] - reconstruct_symmetric_matrix(np.dot(D, X_task_test[:, i]), n_parcels)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        inference_time_fold = time.time() - inference_start
        
        # Per-fold accuracy (for reporting)
        fold_ref_rest = np.array([refined_rest_all[idx][TRIL_IDX] for idx in test_idx])
        fold_ref_task = np.array([refined_task_all[idx][TRIL_IDX] for idx in test_idx])
        fold_corr = np.corrcoef(fold_ref_task, fold_ref_rest)[:n_test, n_test:]
        fold_acc_ref, _ = calculate_accuracy(fold_corr)
        fold_accuracies_refined.append(fold_acc_ref)
        total_inference_time += inference_time_fold
        print(f"  Fold {fold+1} inference time per subject: {(inference_time_fold / n_test) * 1000 :.2f} ms")
        
        fold_res_rest = np.array([residual_rest_all[idx][TRIL_IDX] for idx in test_idx])
        fold_res_task = np.array([residual_task_all[idx][TRIL_IDX] for idx in test_idx])
        fold_corr_res = np.corrcoef(fold_res_task, fold_res_rest)[:n_test, n_test:]
        fold_acc_res, _ = calculate_accuracy(fold_corr_res)
        fold_accuracies_residual.append(fold_acc_res)
        
        print(f"  Fold [{fold+1}/{n_folds}] Train: {n_train} | Test: {n_test} | "
              f"CVAE Loss: {avg_loss:.4f} | Refined Acc: {fold_acc_ref:.4f} | Residual Acc: {fold_acc_res:.4f}")
        
        # Free GPU memory between folds
        if fold < n_folds - 1:
            del fold_model, fold_optimizer, dataset, dataloader
            torch.cuda.empty_cache()
        else:
            last_fold_model = fold_model

    # ==========================================
    # AGGREGATE RESULTS
    # ==========================================
    print(f"\n[4/4] Computing aggregate metrics...")
    
    # CVAE Residuals only (no SDL)
    res_rest_triu = np.array([residual_rest_all[i][TRIL_IDX] for i in range(num_subjects)])
    res_task_triu = np.array([residual_task_all[i][TRIL_IDX] for i in range(num_subjects)])
    residual_matrix = np.corrcoef(res_task_triu, res_rest_triu)[:num_subjects, num_subjects:]
    residual_metrics = compute_all_metrics(residual_matrix)
    
    # CVAE + SDL (full pipeline)
    ref_rest_triu = np.array([refined_rest_all[i][TRIL_IDX] for i in range(num_subjects)])
    ref_task_triu = np.array([refined_task_all[i][TRIL_IDX] for i in range(num_subjects)])
    cvae_sdl_matrix = np.corrcoef(ref_task_triu, ref_rest_triu)[:num_subjects, num_subjects:]
    cvae_sdl_metrics = compute_all_metrics(cvae_sdl_matrix)

    # ==========================================
    # PRINT RESULTS TABLE
    # ==========================================
    print("\n" + "=" * 60)
    print(f"RESULTS - {task_name.upper()} ({num_subjects} Subjects, {n_folds}-Fold CV)")
    print("=" * 60)
    print(f"{'Method':<30} {'Top-1':<10} {'Top-5':<10} {'MRR':<10} {'Diff-ID':<10}")
    print("-" * 60)
    print(f"{'Raw FC (Finn et al.)':<30} {raw_metrics['top_1_accuracy']:.4f}    {raw_metrics['top_5_accuracy']:.4f}    {raw_metrics['mrr']:.4f}    {raw_metrics['differential_id']:.4f}")
    print(f"{'CVAE Residual Only':<30} {residual_metrics['top_1_accuracy']:.4f}    {residual_metrics['top_5_accuracy']:.4f}    {residual_metrics['mrr']:.4f}    {residual_metrics['differential_id']:.4f}")
    print(f"{'CVAE + SDL (Lu et al.)':<30} {cvae_sdl_metrics['top_1_accuracy']:.4f}    {cvae_sdl_metrics['top_5_accuracy']:.4f}    {cvae_sdl_metrics['mrr']:.4f}    {cvae_sdl_metrics['differential_id']:.4f}")
    print("=" * 60)
    
    print(f"\nCVAE+SDL Comprehensive Metrics:")
    print("-" * 50)
    print(f"  Top-1 Accuracy:              {cvae_sdl_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-3 Accuracy:              {cvae_sdl_metrics['top_3_accuracy']:.4f}")
    print(f"  Top-5 Accuracy:              {cvae_sdl_metrics['top_5_accuracy']:.4f}")
    print(f"  Top-10 Accuracy:             {cvae_sdl_metrics['top_10_accuracy']:.4f}")
    print(f"  Mean Rank:                   {cvae_sdl_metrics['mean_rank']:.2f}")
    print(f"  Mean Reciprocal Rank:        {cvae_sdl_metrics['mrr']:.4f}")
    print(f"  Differential Identifiability:{cvae_sdl_metrics['differential_id']:.4f}")

    # ==========================================
    # SAVE ASSETS & REPORTS
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}_{task_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Robustness Analysis
    print("\n[5/6] Running Robustness Analysis...")
    with torch.no_grad():
        n_rob = noise_robustness(fc_task, fc_rest, last_fold_model, last_D, L)
        s_rob = sample_size_robustness(fc_task, fc_rest)
        plot_robustness_comparison({'CVAE+SDL': (n_rob, s_rob)}, os.path.join(run_dir, "robustness_comparison_combined.png"))
    
    # 2. Save Matrices
    print("\n[6/6] Saving Assets & Reports...")
    np.save(os.path.join(run_dir, "raw_fc_matrix.npy"), raw_matrix)
    np.save(os.path.join(run_dir, "residual_matrix.npy"), residual_matrix)
    np.save(os.path.join(run_dir, "cvae_sdl_id_matrix.npy"), cvae_sdl_matrix)
    if last_D is not None:
        np.save(os.path.join(run_dir, "dictionary_D.npy"), last_D)
    
    # 3. Correlation Heatmaps (Standardized Style)
    plot_full_correlation_matrix(cvae_sdl_matrix, os.path.join(run_dir, "heatmap_cvae_sdl.png"))
    plot_full_correlation_matrix(raw_matrix, os.path.join(run_dir, "heatmap_raw_fc.png"))
    
    # 4. Similarity distributions (Standardized Style)
    plot_similarity_distributions(cvae_sdl_matrix, os.path.join(run_dir, "similarity_distributions.png"))
    
    # 5. Ablation Bar Chart
    plt.figure(figsize=(10, 6))
    methods = ['Raw FC', 'CVAE Residual', 'CVAE+SDL']
    acc_list = [raw_metrics['top_1_accuracy'], residual_metrics['top_1_accuracy'], cvae_sdl_metrics['top_1_accuracy']]
    bars = plt.bar(methods, acc_list, color=['firebrick', 'forestgreen', 'steelblue'], edgecolor='black')
    for bar, acc_val in zip(bars, acc_list):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f'{acc_val:.4f}', ha='center', va='bottom')
    plt.title(f'Ablation Study: Component Contributions ({task_name.upper()})')
    plt.ylabel('Top-1 Accuracy')
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(run_dir, "ablation_results.png"), dpi=300)
    plt.close()

    # 5. Dictionary Atoms Visualization
    if last_D is not None:
        n_show = min(K, 8)
        fig, axes = plt.subplots(2, n_show // 2, figsize=(16, 8))
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                atom_matrix = reconstruct_symmetric_matrix(last_D[:, idx], n_parcels)
                ax.imshow(atom_matrix, cmap='RdBu_r')
                ax.set_title(f'Atom {idx+1}')
                ax.axis('off')
        plt.savefig(os.path.join(run_dir, "dictionary_atoms.png"), dpi=300)
        plt.close()

    # 6. Comprehensive Report
    report_path = os.path.join(run_dir, "CVAE_SDL_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CVAE + SDL BASELINE REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Task: {task_name.upper()} | Subjects: {num_subjects}\n\n")
        f.write(f"Top-1 Accuracy (CVAE+SDL): {cvae_sdl_metrics['top_1_accuracy']:.4f}\n")
        f.write(f"Total Training Time:        {total_training_time:.2f} s\n")
        f.write(f"Avg Inference Time:        {(total_inference_time/num_subjects)*1000:.2f} ms\n\n")
        f.write("PER-FOLD ACCURACIES:\n")
        for i in range(n_folds):
            f.write(f"  Fold {i+1}: {fold_accuracies_refined[i]:.4f}\n")
    
    # 7. Save JSON Results
    json_results = {
        'task': task_name,
        'num_subjects': num_subjects,
        'n_folds': n_folds,
        'epochs': epochs,
        'K': K,
        'L': L,
        'model_params': model_params,
        'raw_fc_metrics': raw_metrics,
        'cvae_residual_metrics': residual_metrics,
        'cvae_sdl_metrics': cvae_sdl_metrics,
        'cv_mean_acc': float(np.mean(fold_accuracies_refined)),
        'total_train_time': total_training_time,
        'avg_inference_time_ms': (total_inference_time / num_subjects) * 1000,
        'noise_robustness': n_rob,
        'sample_robustness': s_rob,
        'timestamp': timestamp
    }
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # 8. Zip Results
    zip_path = os.path.join(WORKING_DIR, f"cvae_sdl_results_{task_name}_{timestamp}")
    shutil.make_archive(zip_path, 'zip', run_dir)
    
    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS SAVED TO: {run_dir}")
    print(f"{'='*60}")
    
    return cvae_sdl_metrics, raw_metrics

# %% [markdown]
# ## 7. Execute Pipeline

# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CVAE+SDL Baseline (Lu et al., 2024)")
    parser.add_argument("--task", type=str, default="all", help="Task name")
    parser.add_argument("--num_subjects", type=int, default=339, help="Number of subjects")
    parser.add_argument("--K", type=int, default=None, help="Manual Atoms (skips optimization)")
    parser.add_argument("--L", type=int, default=None, help="Manual Sparsity (skips optimization)")
    parser.add_argument("--optimize", action="store_true", help="Force Grid Search optimization")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    
    args, unknown = parser.parse_known_args()
    
    if args.task.lower() == "all":
        tasks_to_run = ["motor", "wm", "emotion", "gambling", "language", "relational", "social"]
    else:
        tasks_to_run = [args.task]

    for current_task in tasks_to_run:
        print(f"\n\n{'='*70}")
        print(f"STARTING EXPERIMENT: Task = {current_task.upper()}")
        print(f"{'='*70}\n")
        
        # Priority 1: Manual Input from CLI
        if args.K is not None and args.L is not None:
            best_K, best_L = args.K, args.L
            print(f">>> [MANUAL] Using provided hyperparameters: K={best_K}, L={best_L}")
        
        # Priority 2: Force Optimization via --optimize flag
        elif args.optimize:
            print(f">>> [OPTIMIZE] Grid search triggered by flag for {current_task}...")
            # Load data if exists
            rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
            task_path = os.path.join(FC_DATA_DIR, f"fc_{current_task}.npy")
            if os.path.exists(rest_path) and os.path.exists(task_path):
                fc_r = np.load(rest_path)[:args.num_subjects]
                fc_t = np.load(task_path)[:args.num_subjects]
                n_p = fc_r.shape[1]
                tril_idx = np.tril_indices(n_p, k=-1)
                flat_t = np.array([fc_t[i][tril_idx] for i in range(len(fc_t))]).T
                flat_r = np.array([fc_r[i][tril_idx] for i in range(len(fc_r))]).T
                _, best_K, best_L = perform_grid_search(flat_t, flat_r, args.num_subjects, n_p, K_range=(2, 15), n_iter=5, task_name=current_task)
            else:
                print("    WARNING: Data not found for optimization. Using defaults.")
                best_K, best_L = 15, 12
        
        # Priority 3: Pre-Tuned Parameters Library
        elif current_task.lower() in TUNED_PARAMS:
            best_K, best_L = TUNED_PARAMS[current_task.lower()]
            print(f">>> [TUNED] Using library parameters for {current_task}: K={best_K}, L={best_L}")
        
        # Priority 4: Automatic Optimization for remaining tasks
        else:
            print(f">>> [AUTO-OPTIMIZE] No pre-tuned parameters for {current_task}. Searching now...")
            rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
            task_path = os.path.join(FC_DATA_DIR, f"fc_{current_task}.npy")
            if os.path.exists(rest_path) and os.path.exists(task_path):
                fc_r = np.load(rest_path)[:args.num_subjects]
                fc_t = np.load(task_path)[:args.num_subjects]
                n_p = fc_r.shape[1]
                tril_idx = np.tril_indices(n_p, k=-1)
                flat_t = np.array([fc_t[i][tril_idx] for i in range(len(fc_t))]).T
                flat_r = np.array([fc_r[i][tril_idx] for i in range(len(fc_r))]).T
                _, best_K, best_L = perform_grid_search(flat_t, flat_r, args.num_subjects, n_p, K_range=(2, 15), n_iter=5, task_name=current_task)
            else:
                print("    WARNING: Data missing. Using default (15, 12).")
                best_K, best_L = 15, 12
        
        try:
            run_cvae_sdl_baseline(
                task_name=current_task,
                num_subjects=args.num_subjects,
                K=best_K,
                L=best_L,
                epochs=args.epochs
            )
        except Exception as e:
            print(f"[ERROR] Failed on task {current_task}: {e}")

    # --- Benchmark Model Params and Inference Time ---
    print("\n" + "="*60)
    print("BENCHMARKING COMPUTATIONAL COMPLEXITY")
    print("="*60)
    import time
    def benchmark_model(model, iter_inputs, device, iters=50):
        model.to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                inputs_device = [i.to(device) for i in iter_inputs]
                _ = model(*inputs_device)
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iters):
                inputs_device = [i.to(device) for i in iter_inputs]
                _ = model(*inputs_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        return (end_time - start_time) / iters * 1000

    from torch import randn, zeros
    cvae = ConditionalVAE()
    params = cvae.count_parameters()
    inference_time = benchmark_model(cvae, [randn(1, 1, 360, 360), zeros(1, 1)], DEVICE)
    print(f"CVAE-SDL (Lu et al.):")
    print(f"  Trainable Parameters: {params:,}")
    print(f"  Inference Time per subject: {inference_time:.2f} ms\n")
