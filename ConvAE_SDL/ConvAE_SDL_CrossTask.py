import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.linear_model import orthogonal_mp
from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import zipfile

# Aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Kaggle Directory Structure
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
INPUT_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "."
OUTPUT_DIR = os.path.join(WORKING_DIR, "manuscript_results")
SAVE_DIR = os.path.join(WORKING_DIR, "FC_DATA")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Global variables for raw data paths
RAW_REST_DIR = None
RAW_TASK_DIR = None
FC_DATA_DIR = SAVE_DIR

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def setup_environment():
    """Consolidated path discovery and data acquisition logic."""
    global RAW_REST_DIR, RAW_TASK_DIR, FC_DATA_DIR
    
    # Check for Zip & Extract if needed
    has_subjects = False
    for root, dirs, _ in os.walk(WORKING_DIR):
        if "subjects" in dirs:
            has_subjects = True
            break
    
    if not has_subjects:
        for root, _, files in os.walk(INPUT_DIR):
            if "DATA.zip" in files:
                zip_path = os.path.join(root, "DATA.zip")
                print(f">>> Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(WORKING_DIR)
                break
    
    # Locate Raw Data (Searching for 'subjects' folders)
    RAW_REST_DIR = None
    RAW_TASK_DIR = None
    
    for d in [INPUT_DIR, WORKING_DIR]:
        for root, dirs, _ in os.walk(d):
            if "subjects" in dirs:
                if "rest" in root.lower():
                    RAW_REST_DIR = root
                elif "task" in root.lower() or "motor" in root.lower() or "hcp_task" in root.lower():
                    RAW_TASK_DIR = root
                if RAW_REST_DIR is None: RAW_REST_DIR = root
                if RAW_TASK_DIR is None: RAW_TASK_DIR = root
    
    # Locate Pre-calculated FC Data
    fc_dir = SAVE_DIR
    explicit_paths = [
        "/kaggle/input/fc-data",
        "/kaggle/input/fc-data/fc_data"
    ]
    found_explicit = False
    for path in explicit_paths:
        if os.path.exists(os.path.join(path, "fc_rest.npy")):
            fc_dir = path
            found_explicit = True
            break
            
    if not found_explicit:
        for root, _, files in os.walk(INPUT_DIR):
            if "fc_rest.npy" in files:
                fc_dir = root
                break
                
    FC_DATA_DIR = fc_dir

    print(f"--- Environment Verified ---")
    print(f"Device: {DEVICE.upper()}")
    print(f"Raw Rest: {RAW_REST_DIR or 'Not Found'}")
    print(f"Raw Task: {RAW_TASK_DIR or 'Not Found'}")
    print(f"FC Data Path: {fc_dir}")
    print(f"----------------------------\n")

# Run environment setup
setup_environment()

# Constants
BOLD_NAMES = [
    "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL",
    "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR", "tfMRI_WM_RL", "tfMRI_WM_LR",
    "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"
]
TRIL_IDX = np.tril_indices(360, k=-1)

def get_image_ids(name):
    run_ids = [i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code]
    if not run_ids: 
        raise ValueError(f"Found no data for '{name}'")
    return run_ids

def load_single_timeseries(subject_id, bold_run, base_dir, remove_mean=True):
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

# Variables
USE_SYNTHETIC = False
N_SUBJECTS = 339
N_PARCELS = 360
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Hyperparameters for SDL
DEFAULT_K = 15
DEFAULT_L = 12

TUNED_PARAMS = {
    "motor": (14, 12),
    "wm": (15, 13),
    "emotion": (15, 13),
    "gambling": (14, 11),
    "language": (15, 11),
    "relational": (11, 9),
    "social": (15, 9)
}

TASKS = ["social", "motor", "wm", "emotion", "gambling", "language", "relational"]
EVAL_TARGETS = ["rest"] + TASKS

# --- Core Model Architectures ---

class ConvAutoencoder(nn.Module):
    def __init__(self, n_parcels=360):
        super(ConvAutoencoder, self).__init__()
        self.n = n_parcels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if decoded.shape[2:] != (self.n, self.n):
            decoded = F.interpolate(decoded, size=(self.n, self.n), mode='bilinear', align_corners=True)
        return decoded

# --- Sparse Dictionary Learning ---

def omp_sparse_coding(Y, D, L):
    n_samples = Y.shape[1]
    n_atoms = D.shape[1]
    X = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        X[:, i] = orthogonal_mp(D, Y[:, i], n_nonzero_coefs=L)
    return X

def update_dictionary(Y, D, X):
    for k in range(D.shape[1]):
        idx = np.nonzero(X[k, :])[0]
        if len(idx) == 0:
            D[:, k] = np.random.randn(D.shape[0])
            D[:, k] /= np.linalg.norm(D[:, k])
            continue
        residual = Y[:, idx] - np.dot(D, X[:, idx])
        residual += np.outer(D[:, k], X[k, idx])
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        D[:, k] = U[:, 0]
        X[k, idx] = S[0] * Vt[0, :]
    return D, X

def k_svd(Y, K, L, n_iter=10, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    m, n = Y.shape
    D = np.random.randn(m, K)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    for _ in range(n_iter):
        X = omp_sparse_coding(Y, D, L)
        D, X = update_dictionary(Y, D, X)
    return D, X

def reconstruct_symmetric_matrix(tri_elements, n=360):
    tril_idx = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n))
    matrix[tril_idx] = tri_elements
    matrix += matrix.T
    np.fill_diagonal(matrix, 1.0)
    return matrix

# --- Metrics ---

def calculate_accuracy(corr_matrix):
    if corr_matrix is None or corr_matrix.size == 0:
        return 0.0
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0)
    n = corr_matrix.shape[0]
    correct = sum(1 for i in range(n) if np.argmax(corr_matrix[i, :]) == i)
    return correct / n

# --- Data Generation ---

def generate_synthetic_fc(n_subjects, n_parcels, seed_base=0):
    np.random.seed(seed_base)
    fc = np.zeros((n_subjects, n_parcels, n_parcels))
    for i in range(n_subjects):
        shared = np.random.randn(n_parcels, n_parcels) * 0.3
        shared = (shared + shared.T) / 2
        unique = np.random.randn(n_parcels, n_parcels) * 0.5
        unique = (unique + unique.T) / 2
        mat = shared + unique
        np.fill_diagonal(mat, 1.0)
        mat = np.clip(mat, -1, 1)
        fc[i] = mat
    return fc

# --- Main Script ---

def main():
    print(">>> Initializing Data...")
    
    fc_db = None
    fc_target_dict = {}
    
    # Try loading real data first
    if not USE_SYNTHETIC and os.path.exists(FC_DATA_DIR):
        print(f"Loading pre-computed FC matrices from: {FC_DATA_DIR}")
        
        # Load Rest
        rest_path = os.path.join(FC_DATA_DIR, "fc_rest.npy")
        if os.path.exists(rest_path):
            full_rest = np.load(rest_path)
            if full_rest.shape[0] > N_SUBJECTS:
                full_rest = full_rest[:N_SUBJECTS]
            fc_db = full_rest
            fc_target_dict["rest"] = full_rest
            print(f"  [OK] Loaded Rest FC: {fc_db.shape}")
            
        # Load Tasks
        for t in TASKS:
            task_path = os.path.join(FC_DATA_DIR, f"fc_{t}.npy")
            if os.path.exists(task_path):
                full_task = np.load(task_path)
                if full_task.shape[0] > N_SUBJECTS:
                    full_task = full_task[:N_SUBJECTS]
                fc_target_dict[t] = full_task
                print(f"  [OK] Loaded Task FC ({t}): {full_task.shape}")
                
    # If not all required data was loaded, generate synthetics or extract from RAW
    if fc_db is None or len(fc_target_dict) <= len(TASKS):
        print("\n>>> Real data missing or incomplete.")
        
        # We can attempt to generate them if RAW_DIRs exist, but to be safe and 
        # keep the cross-task fast for demonstration, we fall back to synthetic.
        # If running in a true environment, generation logic from `generate_fc_for_task` applies.
        
        if RAW_REST_DIR and RAW_TASK_DIR and not USE_SYNTHETIC:
            print(">>> RAW directories found, generating missing FC matrices...")
            subjects_path = os.path.join(RAW_REST_DIR, "subjects")
            if os.path.exists(subjects_path):
                all_subs = sorted([d for d in os.listdir(subjects_path) if os.path.isdir(os.path.join(subjects_path, d))])
                if len(all_subs) > N_SUBJECTS:
                    all_subs = all_subs[:N_SUBJECTS]
                
                if fc_db is None:
                    fc_r, _ = generate_fc_for_task("rest", all_subs, RAW_REST_DIR)
                    np.save(os.path.join(FC_DATA_DIR, "fc_rest.npy"), fc_r)
                    fc_db = fc_r
                    fc_target_dict["rest"] = fc_r
                
                for t in TASKS:
                    if t not in fc_target_dict:
                        fc_t, _ = generate_fc_for_task(t, all_subs, RAW_TASK_DIR)
                        np.save(os.path.join(FC_DATA_DIR, f"fc_{t}.npy"), fc_t)
                        fc_target_dict[t] = fc_t
                        
        if fc_db is None or len(fc_target_dict) <= len(TASKS):
            print(">>> Still missing data, falling back to synthetic generation...")
            if fc_db is None:
                fc_db = generate_synthetic_fc(N_SUBJECTS, N_PARCELS, seed_base=100)
                fc_target_dict["rest"] = generate_synthetic_fc(N_SUBJECTS, N_PARCELS, seed_base=200)
            
            for t in TASKS:
                if t not in fc_target_dict:
                    fc_target_dict[t] = generate_synthetic_fc(N_SUBJECTS, N_PARCELS, seed_base=hash(t) % 1000)

    # Initialize results table
    cross_task_results = pd.DataFrame(index=TASKS, columns=EVAL_TARGETS, dtype=object)

    for train_task in TASKS:
        print(f"\n{'='*50}")
        print(f"Training on Task: {train_task.upper()}")
        print(f"{'='*50}")
        
        # 1. Prepare training data
        fc_train = fc_target_dict[train_task]
        train_tensor = torch.tensor(fc_train[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
        loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=BATCH_SIZE, shuffle=True)
        
        # 2. Train ConvAE
        print("  -> Training ConvAE...")
        model = ConvAutoencoder(N_PARCELS).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(EPOCHS):
            for batch, _ in loader:
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
        
        # 3. Extract Residuals
        print("  -> Extracting Residuals...")
        model.eval()
        with torch.no_grad():
            reconstructed_all = model(train_tensor).cpu().numpy().squeeze()
        residuals = fc_train - reconstructed_all
        
        # 4. Train SDL Dictionary
        print("  -> Training SDL Dictionary...")
        K, L = TUNED_PARAMS.get(train_task, (DEFAULT_K, DEFAULT_L))
        n_tri = int(N_PARCELS * (N_PARCELS - 1) / 2)
        tril_idx = np.tril_indices(N_PARCELS, k=-1)
        
        Y_train = np.zeros((n_tri, N_SUBJECTS))
        for i in range(N_SUBJECTS):
            Y_train[:, i] = residuals[i][tril_idx]
            
        D, _ = k_svd(Y_train, K, L, n_iter=10, random_state=42)
        
        # 5. Evaluate on all target conditions
        print(f"  -> Evaluating cross-task generalization...")
        for eval_target in EVAL_TARGETS:
            if train_task == eval_target:
                cross_task_results.loc[train_task, eval_target] = "--"
                continue
                
            fc_eval = fc_target_dict[eval_target]
            eval_tensor = torch.tensor(fc_eval[:, np.newaxis, :, :], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                recon_eval = model(eval_tensor).cpu().numpy().squeeze()
            
            res_eval = fc_eval - recon_eval
            Y_eval = np.zeros((n_tri, N_SUBJECTS))
            for i in range(N_SUBJECTS):
                Y_eval[:, i] = res_eval[i][tril_idx]
                
            # Sparse coding on test set with frozen dictionary
            X_eval = omp_sparse_coding(Y_eval, D, L)
            sdl_recon = np.dot(D, X_eval).T
            
            # Refine fingerprints
            refined_eval = np.zeros_like(fc_eval)
            for i in range(N_SUBJECTS):
                refined_eval[i] = res_eval[i] - reconstruct_symmetric_matrix(sdl_recon[i], N_PARCELS)
                
            # Compute accuracy
            # If target is Rest, we cannot match Rest vs Rest (gives 100%). We match Refined Rest vs Raw Training Task.
            # If target is a Task, we match Refined Task vs Raw Rest (standard pipeline).
            flat_eval = refined_eval.reshape(N_SUBJECTS, -1)
            if eval_target == "rest":
                flat_db = fc_train.reshape(N_SUBJECTS, -1)
            else:
                flat_db = fc_db.reshape(N_SUBJECTS, -1)
                
            corr = np.corrcoef(flat_eval, flat_db)[:N_SUBJECTS, N_SUBJECTS:]
            acc = calculate_accuracy(corr) * 100
            
            cross_task_results.loc[train_task, eval_target] = f"{acc:.2f}%"
            print(f"     Target: {eval_target.upper():<12} | Top-1 Accuracy: {acc:.2f}%")

    print("\n\n" + "="*80)
    print("Table II: Cross-Task Generalization: Identification Accuracy (%)")
    print("="*80)
    print(cross_task_results.to_string())
    print("="*80)
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "cross_task_generalization.csv")
    cross_task_results.to_csv(output_path)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    main()
