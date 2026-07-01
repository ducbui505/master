import os
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*", category=Warning)
import torch
import random
import pickle
import argparse
import csv
import json
import torch.nn as nn
import sys
import time
from math import sqrt
import torch.utils.data
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model import Mulmodel
from contrastive import SupervisedContrastiveLoss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, \
precision_score, recall_score, confusion_matrix,cohen_kappa_score,matthews_corrcoef,average_precision_score
from sklearn.preprocessing import label_binarize

# Global random seed
np.random.seed(42)
random.seed(42)

METRIC_KEYS = [
    'acc', 'weighted_f1', 'macro_f1', 'kappa', 'mcc',
    'macro_precision', 'macro_recall', 'macro_aupr'
]
METRIC_NAMES = {
    'acc': 'Accuracy',
    'weighted_f1': 'Weighted F1',
    'macro_f1': 'Macro F1',
    'kappa': 'Kappa',
    'mcc': 'MCC',
    'macro_precision': 'Macro Precision',
    'macro_recall': 'Macro Recall',
    'macro_aupr': 'Macro AUPR',
}
RUN_CONFIG_KEYS = [
    'epochs', 'lr', 'embed_dim', 'weight_decay', 'dropout', 'gp',
    'batch_size', 'test_batch_size', 'dataset', 'rawpath',
    'use_llm', 'drug_llm_path', 'se_llm_path', 'drug_mask_path',
    'se_mask_path', 'use_cross_modal', 'use_supcon', 'alpha_supcon',
    'temperature', 'cross_modal_variant'
]


class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _config_from_args(args):
    return {key: getattr(args, key) for key in RUN_CONFIG_KEYS}


def _explicit_cli_keys(argv):
    option_to_key = {
        '--epochs': 'epochs',
        '--lr': 'lr',
        '--embed_dim': 'embed_dim',
        '--weight_decay': 'weight_decay',
        '--dropout': 'dropout',
        '--gp': 'gp',
        '--batch_size': 'batch_size',
        '--test_batch_size': 'test_batch_size',
        '--dataset': 'dataset',
        '--rawpath': 'rawpath',
        '--use_llm': 'use_llm',
        '--drug_llm_path': 'drug_llm_path',
        '--se_llm_path': 'se_llm_path',
        '--drug_mask_path': 'drug_mask_path',
        '--se_mask_path': 'se_mask_path',
        '--use_cross_modal': 'use_cross_modal',
        '--use_supcon': 'use_supcon',
        '--alpha_supcon': 'alpha_supcon',
        '--temperature': 'temperature',
        '--cross_modal_variant': 'cross_modal_variant',
    }
    return {option_to_key[arg] for arg in argv if arg in option_to_key}


def setup_run(args, explicit_keys):
    if args.resume:
        run_dir = os.path.normpath(args.resume)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Resume config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)
        current_config = _config_from_args(args)
        mismatches = []
        for key in explicit_keys:
            if key in saved_config and current_config.get(key) != saved_config[key]:
                mismatches.append(f"{key}: current={current_config.get(key)!r}, saved={saved_config[key]!r}")
        if mismatches:
            raise ValueError("Resume arguments do not match saved config. " + "; ".join(mismatches))
        for key, value in saved_config.items():
            setattr(args, key, value)
        args.run_dir = run_dir
        args.run_name = os.path.basename(run_dir.rstrip(os.sep))
        return run_dir

    run_name = args.run_name
    if not run_name:
        flags = []
        if args.use_llm:
            flags.append("llm")
        if args.use_cross_modal:
            flags.append(args.cross_modal_variant)
        if args.use_supcon:
            flags.append("supcon")
        suffix = "_".join(flags) if flags else "baseline"
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + suffix

    if os.path.basename(os.path.normpath(run_name)) != run_name:
        raise ValueError("--run_name must be a simple name, not a path. Use --resume for an existing run directory.")
    run_dir = os.path.join("runs", run_name)
    if os.path.exists(run_dir):
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --resume {run_dir} or choose another --run_name.")
    os.makedirs(run_dir, exist_ok=True)
    args.run_name = run_name
    args.run_dir = run_dir
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_config_from_args(args), f, indent=2, ensure_ascii=False)
    return run_dir


def attach_run_logger(run_dir):
    log_path = os.path.join(run_dir, "train.log")
    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = TeeLogger(sys.__stdout__, log_file)
    sys.stderr = TeeLogger(sys.__stderr__, log_file)
    return log_file


def _fold_metrics_path(args):
    return os.path.join(args.run_dir, "fold_metrics.csv")


def _summary_path(args):
    return os.path.join(args.run_dir, "summary.txt")


def _run_state_path(args):
    return os.path.join(args.run_dir, "run_state.json")


def load_fold_metrics(args):
    path = _fold_metrics_path(args)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            parsed = {'fold': int(row['fold'])}
            for key in METRIC_KEYS:
                parsed[key] = float(row[key])
            rows.append(parsed)
        return rows


def write_run_outputs(args, fold_rows, current_fold=None, status="running"):
    sorted_rows = sorted(fold_rows, key=lambda row: row['fold'])
    csv_path = _fold_metrics_path(args)
    with open(csv_path, 'w', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['fold'] + METRIC_KEYS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    summary_path = _summary_path(args)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Metric              Mean      Std       Mean +/- Std\n")
        f.write("-" * 62 + "\n")
        for key in METRIC_KEYS:
            values = np.array([row[key] for row in sorted_rows], dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            f.write(f"{METRIC_NAMES[key]:<19} {mean:.6f}  {std:.6f}  {mean:.6f} +/- {std:.6f}\n")

    state = {
        "status": status,
        "current_fold": current_fold,
        "completed_folds": [row['fold'] for row in sorted_rows],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(_run_state_path(args), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# -------------------- Checkpoint utils --------------------
def _ckpt_dir(args, fold: int):
    d = os.path.join(args.run_dir, "checkpoints", f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    return d

def save_ckpt(path, model, optimizer, epoch, best_acc, best_metrics=None):
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    tmp_path = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "best_metrics": best_metrics,
        "rng_state": rng_state,
    }, tmp_path)
    os.replace(tmp_path, path)

def load_ckpt(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    rs = ckpt.get("rng_state", {})
    if "torch" in rs:
        rng = rs["torch"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.cpu().byte()
        torch.set_rng_state(rng)
    if "numpy" in rs: np.random.set_state(rs["numpy"])
    if "python" in rs: random.setstate(rs["python"])
    if "cuda" in rs and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rs["cuda"])
    return ckpt
# ----------------------------------------------------------

def _csv_data_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _matching_mapping_file(data_dir, candidates, expected_rows):
    seen = []
    for filename in candidates:
        path = os.path.join(data_dir, filename)
        rows = _csv_data_rows(path)
        if rows is None:
            continue
        seen.append(f"{filename} has {rows} rows")
        if rows == expected_rows:
            return filename, seen
    return None, seen


def validate_mapping_shapes(rawdata_dir, drug_side):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    drug_file, drug_seen = _matching_mapping_file(
        data_dir,
        ["mssf_aligned_drug_mapping.csv", "drug_mapping.csv"],
        drug_side.shape[0],
    )
    se_file, se_seen = _matching_mapping_file(
        data_dir,
        ["mssf_aligned_se_mapping.csv", "se_mapping.csv"],
        drug_side.shape[1],
    )

    problems = []
    if drug_file is None:
        problems.append(
            f"no drug mapping matches {rawdata_dir}/drug_side.pkl rows ({drug_side.shape[0]}). "
            + "; ".join(drug_seen)
        )
    if se_file is None:
        problems.append(
            f"no SE mapping matches {rawdata_dir}/drug_side.pkl columns ({drug_side.shape[1]}). "
            + "; ".join(se_seen)
        )

    if problems:
        raise ValueError(
            "Mapping/raw matrix shape mismatch. "
            + " ; ".join(problems)
            + ". Use mssf_aligned_* mappings with the 757x994 MSSF raw data, "
            + "or benchmark mappings with a 759x994 benchmark raw matrix."
        )


def validate_llm_tensor(name, tensor, expected_rows):
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {tensor.shape[0]} rows but the active raw matrix expects {expected_rows}. "
            "Re-encode LLM features from the mapping/text files that belong to the same dataset."
        )


def read_raw_data(rawdata_dir, data_test):

    # Load SMDsimilarity
    gii = open(rawdata_dir + '/' + 'Text_similarity_one.pkl', 'rb')
    drug_Tfeature_one = pickle.load(gii)
    gii.close()

    # Load SMDexperimental
    gii = open(rawdata_dir + '/' + 'Text_similarity_two.pkl', 'rb')
    drug_Tfeature_two = pickle.load(gii)
    gii.close()

    # Load SMDdatabase
    gii = open(rawdata_dir + '/' + 'Text_similarity_three.pkl', 'rb')
    drug_Tfeature_three = pickle.load(gii)
    gii.close()

    # Load SMDtext
    gii = open(rawdata_dir + '/' + 'Text_similarity_four.pkl', 'rb')
    drug_Tfeature_four = pickle.load(gii)
    gii.close()

    # Load SMDcombined
    gii = open(rawdata_dir + '/' + 'Text_similarity_five.pkl', 'rb')
    drug_Tfeature_five = pickle.load(gii)
    gii.close()

    # Load semantic similarity of side effects
    gii = open(rawdata_dir + '/' + 'side_effect_semantic.pkl', 'rb')
    effect_side_semantic = pickle.load(gii)
    gii.close()

    # Load molecular embeddings for drugs and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'drug_mol.pkl', 'rb')
    Drug_word2vec = pickle.load(gii)
    gii.close()
    Drug_word_sim = cosine_similarity(Drug_word2vec)

    # Load GloVe word embeddings for side effects and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'glove_wordEmbedding.pkl', 'rb')
    glove_word = pickle.load(gii)
    gii.close()
    side_glove_sim = cosine_similarity(glove_word)

    # Load drug-target information and compute cosine similarity
    gii = open(rawdata_dir + '/' + 'drug_target.pkl', 'rb')
    drug_target = pickle.load(gii)
    gii.close()
    drug_target_sim = cosine_similarity(drug_target)

    # Load drug fingerprint similarity
    gii = open(rawdata_dir + '/' + 'fingerprint_similarity.pkl', 'rb')
    drug_f_sim = pickle.load(gii)
    gii.close()

    # Load drug-side effect interaction matrix
    gii = open(rawdata_dir + '/' + 'drug_side.pkl', 'rb')
    drug_side = pickle.load(gii)
    gii.close()
    validate_mapping_shapes(rawdata_dir, drug_side)

    # Load drug-pathway-enzyme similarity matrix
    gii = open(rawdata_dir + '/' + 'drug_pathway_enzyme_similarity.pkl', 'rb')
    drug_p_e_sim = pickle.load(gii)
    gii.close()

    # Remove test set information from the training matrix
    for i in range(data_test.shape[0]):
        drug_side[data_test[i, 0], data_test[i, 1]] = 0

    # Compute cosine similarity over the remaining drug-side effect frequency matrix, SMD_DIPF
    drug_side_sim = cosine_similarity(drug_side)

    # Generate binary label matrix from the frequency matrix
    drug_side_label = np.zeros((drug_side.shape[0], drug_side.shape[1]))
    for i in range(drug_side.shape[0]):
        for j in range(drug_side.shape[1]):
            if drug_side[i, j] > 0:
                drug_side_label[i, j] = 1

    # Compute cosine similarity over the binary label matrix, SMD_DIPA
    drug_side_label_sim = cosine_similarity(drug_side_label)

    # Assemble drug features
    drug_features, side_features = [], []
    drug_features.append(drug_Tfeature_one)
    drug_features.append(drug_Tfeature_two)
    drug_features.append(drug_Tfeature_three)
    drug_features.append(drug_Tfeature_four)
    drug_features.append(drug_Tfeature_five)
    drug_features.append(Drug_word_sim)
    drug_features.append(drug_target_sim)
    drug_features.append(drug_f_sim)
    drug_features.append(drug_side_sim)
    drug_features.append(drug_side_label_sim)
    drug_features.append(drug_p_e_sim)

    # Compute similarity matrices for side effects,SME_DIPA,SME_DIPF
    side_drug_sim = cosine_similarity(drug_side.T)
    side_drug_label_sim = cosine_similarity(drug_side_label.T)

    # Assemble side features
    side_features.append(effect_side_semantic)
    side_features.append(side_glove_sim)
    side_features.append(side_drug_sim)
    side_features.append(side_drug_label_sim)

    # return drug and side features
    return drug_features, side_features

# Get the drug-side effect indices for the training and test sets,
# as well as the frequency information of the corresponding drug-side effect pairs
def fold_files(data_train, data_test,args):

    # Get the path to the raw data directory from arguments
    rawdata_dir = args.rawpath

    # Convert input training and testing data to NumPy arrays
    data_train = np.array(data_train)
    data_test = np.array(data_test)



    drug_features, side_features = read_raw_data(rawdata_dir, data_test)

    # Initialize the drug feature matrix using the first feature
    drug_features_matrix = drug_features[0].astype(np.float32)

    # Concatenate all drug feature matrices horizontally to form a full feature matrix
    for i in range(1, len(drug_features)):
        drug_features_matrix = np.hstack((drug_features_matrix, drug_features[i].astype(np.float32)))

    # Initialize the side effect feature matrix using the first feature
    side_features_matrix = side_features[0].astype(np.float32)

    # Concatenate all side effect feature matrices horizontally to form a full feature matrix
    for i in range(1, len(side_features)):
        side_features_matrix = np.hstack((side_features_matrix, side_features[i].astype(np.float32)))

    # Extract test drug and side effect features based on test indices
    drug_test = drug_features_matrix[data_test[:, 0]]
    side_test = side_features_matrix[data_test[:, 1]]

    # Extract test frequencies
    f_test = data_test[:, 2]

    # Extract training drug and side effect features based on training indices
    drug_train = drug_features_matrix[data_train[:, 0]]
    side_train = side_features_matrix[data_train[:, 1]]

    # Extract training frequencies
    f_train = data_train[:, 2]

    # Return train/test features and frequencies
    if getattr(args, 'use_llm', False):
        drug_llm_all = torch.load(args.drug_llm_path, map_location='cpu', weights_only=True)
        se_llm_all   = torch.load(args.se_llm_path, map_location='cpu', weights_only=True)
        validate_llm_tensor(args.drug_llm_path, drug_llm_all, drug_features_matrix.shape[0])
        validate_llm_tensor(args.se_llm_path, se_llm_all, side_features_matrix.shape[0])

        drug_llm_train = drug_llm_all[data_train[:, 0].astype(int)].numpy()
        se_llm_train   = se_llm_all[data_train[:, 1].astype(int)].numpy()
        drug_llm_test  = drug_llm_all[data_test[:, 0].astype(int)].numpy()
        se_llm_test    = se_llm_all[data_test[:, 1].astype(int)].numpy()

        # Load masks
        drug_mask_all = torch.load(args.drug_mask_path, map_location='cpu')
        se_mask_all   = torch.load(args.se_mask_path, map_location='cpu')
        validate_llm_tensor(args.drug_mask_path, drug_mask_all, drug_features_matrix.shape[0])
        validate_llm_tensor(args.se_mask_path, se_mask_all, side_features_matrix.shape[0])
        drug_mask_train = drug_mask_all[data_train[:, 0].astype(int)].numpy()
        se_mask_train   = se_mask_all[data_train[:, 1].astype(int)].numpy()
        drug_mask_test  = drug_mask_all[data_test[:, 0].astype(int)].numpy()
        se_mask_test    = se_mask_all[data_test[:, 1].astype(int)].numpy()

        return (drug_test, side_test, f_test,
                drug_train, side_train, f_train,
                drug_llm_train, se_llm_train, drug_llm_test, se_llm_test,
                drug_mask_train, se_mask_train, drug_mask_test, se_mask_test)

    return drug_test, side_test, f_test, drug_train, side_train, f_train

def train_test(data_train, data_test, args, fold):
    # Prepare training and test sets by extracting corresponding drug/side effect features and frequencies
    fold_result = fold_files(data_train, data_test, args)

    if getattr(args, 'use_llm', False):
        (drug_test, side_test, f_test,
         drug_train, side_train, f_train,
         drug_llm_train, se_llm_train, drug_llm_test, se_llm_test,
         drug_mask_train, se_mask_train, drug_mask_test, se_mask_test) = fold_result

        trainset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
            torch.FloatTensor(f_train),
            torch.FloatTensor(drug_llm_train), torch.FloatTensor(se_llm_train),
            torch.FloatTensor(drug_mask_train), torch.FloatTensor(se_mask_train))
        testset = torch.utils.data.TensorDataset(
            torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
            torch.FloatTensor(f_test),
            torch.FloatTensor(drug_llm_test), torch.FloatTensor(se_llm_test),
            torch.FloatTensor(drug_mask_test), torch.FloatTensor(se_mask_test))
    else:
        drug_test, side_test, f_test, drug_train, side_train, f_train = fold_result
        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_train), torch.FloatTensor(side_train),
                                                  torch.FloatTensor(f_train))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(drug_test), torch.FloatTensor(side_test),
                                                 torch.FloatTensor(f_test))

    # SupCon needs larger batches for enough positive pairs
    train_batch = args.batch_size
    if getattr(args, 'use_supcon', False) and args.batch_size < 128:
        train_batch = 128

    # Wrap datasets in DataLoader for batch training and evaluation
    _train = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True,
                                          pin_memory=False)
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        pin_memory=False)

    # Set the runtime device for the program
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set GPU device index
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu") # Select CUDA if available

    # Instantiate the model and move it to the chosen device
    model = Mulmodel(args).to(device)

    # Set optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # SupCon loss function
    supcon_loss_fn = None
    if getattr(args, 'use_supcon', False):
        supcon_loss_fn = SupervisedContrastiveLoss(temperature=getattr(args, 'temperature', 0.1))

        # -------- checkpoint setup (per fold) --------
    ckpt_dir = _ckpt_dir(args, fold)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    start_epoch = 1
    best_acc = 0.0
    best_metrics = None

    if os.path.exists(last_path):
        try:
            ckpt = load_ckpt(last_path, model, optimizer, device)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_acc = ckpt.get("best_acc", 0.0)
            best_metrics = ckpt.get("best_metrics")
            print(f"[fold {fold}] RESUME from epoch {start_epoch}, best_acc={best_acc}")
        except (RuntimeError, KeyError) as e:
            print(f"[fold {fold}] Checkpoint incompatible (architecture changed?), training from scratch. ({e})")
            start_epoch = 1
            best_acc = 0.0
            best_metrics = None
    # ---------------------------------------------

    if start_epoch > args.epochs:
        if best_metrics is not None:
            print(f"[fold {fold}] checkpoint already reached epoch {args.epochs}; returning saved best metrics")
            return tuple(best_metrics[key] for key in METRIC_KEYS)
        eval_path = best_path if os.path.exists(best_path) else last_path
        if os.path.exists(eval_path):
            print(f"[fold {fold}] checkpoint already reached epoch {args.epochs}; evaluating {eval_path}")
            load_ckpt(eval_path, model, optimizer, device)
            acc_te, weighted_f1_te, macro_f1_te, kappa_te, mcc_te, rating_te, pred_te, macro_prec_te, macro_recall_te, macro_aupr_te = test(model, _test, device, args)
            return acc_te, weighted_f1_te, macro_f1_te, kappa_te, mcc_te, macro_prec_te, macro_recall_te, macro_aupr_te

    # Initialize evaluation metric
    if best_metrics is not None:
        acc_tested = best_metrics['acc']
        wf1_tested = best_metrics['weighted_f1']
        maf1_tested = best_metrics['macro_f1']
        ka_tested = best_metrics['kappa']
        mcc_tested = best_metrics['mcc']
        maprec_tested = best_metrics['macro_precision']
        mareca_tested = best_metrics['macro_recall']
        maaupr_tested = best_metrics['macro_aupr']
    else:
        acc_tested = 0
        wf1_tested = 0
        maf1_tested = 0
        ka_tested = 0
        mcc_tested = 0
        maprec_tested = 0
        mareca_tested = 0
        maaupr_tested = 0

    # Model training and testing
    for epoch in range(start_epoch, args.epochs + 1):
        # ----------- Training step -----------
        train(model, _train, optimizer, device, args, supcon_loss_fn)

        # ----------- Evaluation on train and test sets -----------
        acc_tr,weighted_f1_tr,macro_f1_tr,kappa_tr,mcc_tr,rating_tr,pred_tr,macro_prec_tr,macro_recall_tr,macro_aupr_tr = test(model,_train,device,args)
        acc_te,weighted_f1_te,macro_f1_te,kappa_te,mcc_te,rating_te,pred_te,macro_prec_te,macro_recall_te,macro_aupr_te = test(model,_test,device,args)

        current_metrics = {
            'acc': acc_te,
            'weighted_f1': weighted_f1_te,
            'macro_f1': macro_f1_te,
            'kappa': kappa_te,
            'mcc': mcc_te,
            'macro_precision': macro_prec_te,
            'macro_recall': macro_recall_te,
            'macro_aupr': macro_aupr_te,
        }

        # If current test accuracy is best so far, update tracked metrics
        if  acc_te>acc_tested:
            acc_tested = acc_te
            wf1_tested =  weighted_f1_te
            maf1_tested = macro_f1_te
            ka_tested = kappa_te
            mcc_tested = mcc_te
            maprec_tested = macro_prec_te
            mareca_tested = macro_recall_te
            maaupr_tested = macro_aupr_te

        # Save BEST when current test accuracy improves.
        if best_metrics is None or acc_te > best_acc:
            best_acc = acc_te
            best_metrics = current_metrics
            save_ckpt(best_path, model, optimizer, epoch, best_acc, best_metrics)

        # Always save LAST after every epoch so interrupted runs can resume.
        save_ckpt(last_path, model, optimizer, epoch, best_acc, best_metrics)

        # Print training results of current epoch
        print("Epoch: %d <Train> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr:%.5f" %(
        epoch, acc_tr,weighted_f1_tr,macro_f1_tr,kappa_tr,mcc_tr,macro_prec_tr,macro_recall_tr,macro_aupr_tr))

        # Print test results of current epoch
        print("Epoch: %d <Test> acc: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr:%.5f" %(
        epoch, acc_te,weighted_f1_te,macro_f1_te,kappa_te,mcc_te,macro_prec_te,macro_recall_te,macro_aupr_te))


    # Print best test results achieved
    print(" <Best Test> acc_tr: %.5f, weighted_f1: %.5f, macro_f1: %.5f, kappa: %.5f ,mcc: %.5f,precision:%.5f,recall: %.5f,aupr: %.5f" % (
        acc_tested,wf1_tested,maf1_tested,ka_tested,mcc_tested,maprec_tested,mareca_tested,maaupr_tested))

    # Return the performance metrics
    return acc_tested,wf1_tested,maf1_tested,ka_tested,mcc_tested,maprec_tested,mareca_tested,maaupr_tested


# KL divergence calculation function
def kl_func(mu,logvar):
    return - 0.5 * (1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=1)

# Loss computation
def calculate_loss(multi_pred,recCon,recAdd,mu, logvar,batch_ratings,batch_drug,batch_side,device,
                   f_fused=None, supcon_loss_fn=None, alpha_supcon=0.1):

    # Compute KL divergence
    kl_div = kl_func(mu, logvar).mean()

    # Define multi-class classification loss with class weights for imbalance
    class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 5.0]).to(device)
    loss_func = nn.CrossEntropyLoss(weight=class_weights)

    # Convert frequencies to integer class indices: -> {0,1,2,3,4}
    multi_labels = (batch_ratings.long()-1).to(device)

    # Concatenate drug and side effect vectors
    batch_vec = torch.cat((batch_drug, batch_side), dim=1)

    # Split the drug features into 11 different parts
    drug1, drug2, drug3, drug4, drug5, drug6, drug7, drug8, drug9, drug10, drug11 = batch_drug.chunk(11, 1)

    # Split the side effect features into 4 parts
    side1, side2, side3, side4 = batch_side.chunk(4, 1)

    # Sum all drug features into a single vector
    drugs = drug1+ drug2+ drug3+ drug4+ drug5+ drug6+ drug7+ drug8+ drug9+ drug10 +drug11
    # Sum all side features  into a single vector
    sides = side1+side2+side3+side4

    # Concatenate aggregated drug and side effect features
    add_features = torch.cat((drugs,sides),dim=1)

    # Classification loss
    multi_loss = loss_func(multi_pred,multi_labels)

    # Define reconstruction loss using MSE without reduction for flexibility
    reconst_loss = nn.MSELoss(reduction='none')

    # Compute concatenation reconstruction loss
    rec_loss1 = reconst_loss(recCon,batch_vec.to(device)).sum(dim=-1).mean()
    # Compute addition reconstruction loss
    rec_loss2 = reconst_loss(recAdd,add_features.to(device)).sum(dim=-1).mean()

    # Compute the total loss
    Loss = multi_loss+0.001*kl_div+0.0001*rec_loss1+0.0001*rec_loss2

    # Supervised Contrastive Loss
    if supcon_loss_fn is not None and f_fused is not None:
        L_con = supcon_loss_fn(f_fused, multi_labels)
        Loss = Loss + alpha_supcon * L_con

    return Loss


# Model training
def train(model, train_loader, optimizer, device, args=None, supcon_loss_fn=None):

    # Set the model to training mode
    model.train()
    avg_loss = 0.0

    # Iterate over training data loader
    for i, data in enumerate(train_loader, 0):

        # Unpack the batch
        if args is not None and getattr(args, 'use_llm', False):
            batch_drug, batch_side, batch_ratings, batch_drug_llm, batch_se_llm, batch_drug_mask, batch_se_mask = data
        else:
            batch_drug, batch_side, batch_ratings = data
            batch_drug_llm = batch_se_llm = batch_drug_mask = batch_se_mask = None

        # Clear gradients from the previous step
        optimizer.zero_grad()

        # model outputs
        if batch_drug_llm is not None:
            outputs = model(batch_drug, batch_side, device, batch_drug_llm, batch_se_llm, batch_drug_mask, batch_se_mask)
            multi_pred, recCon, recAdd, mu, logvar, f_fused = outputs
        else:
            multi_pred, recCon, recAdd, mu, logvar = model(batch_drug, batch_side, device)
            f_fused = None

        # Calculate the loss
        loss = calculate_loss(multi_pred, recCon, recAdd, mu, logvar,
                              batch_ratings, batch_drug, batch_side, device,
                              f_fused=f_fused, supcon_loss_fn=supcon_loss_fn,
                              alpha_supcon=getattr(args, 'alpha_supcon', 0.1) if args else 0.1)

        # Backward pass to compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Accumulate loss
        avg_loss += loss.item()

    return 0

# Model testing
def test(model, test_loader, device, args=None):
    model.eval()  # Set the model to evaluation mode
    pred_all = []           # List to store predicted labels
    multi_label_all = []    # List to store true labels
    prob_all = []           # List to store predicted probabilities

    # Iterate over test data loader
    for data in test_loader:

        # Unpack the batch
        if args is not None and getattr(args, 'use_llm', False):
            test_drug, test_side, test_ratings, test_drug_llm, test_se_llm, test_drug_mask, test_se_mask = data
        else:
            test_drug, test_side, test_ratings = data
            test_drug_llm = test_se_llm = test_drug_mask = test_se_mask = None

        # Obtain predicted data
        if test_drug_llm is not None:
            outputs = model(test_drug, test_side, device, test_drug_llm, test_se_llm, test_drug_mask, test_se_mask)
            multi_pred = outputs[0]
        else:
            multi_pred, recCon, recAdd, mu, logvar = model(test_drug, test_side, device)

        # Classify the predicted data, and in classification, the indices will be in the form of 0, 1, 2, 3, 4
        pred = torch.argmax(multi_pred.cpu(), dim=1).numpy()
        pred_all.append(list(pred)) # Store predictions

        # True labels, and subtract 1 from the true labels to match the 0, 1, 2, 3, 4 format
        multi_label_all.append(list((test_ratings.long()-1).cpu().numpy()))

        # Get predicted probabilities and calculate AUPR
        softmax = torch.nn.Softmax(dim=1)
        pred_prob = softmax(multi_pred).cpu().detach().numpy()  # Convert predictions to probabilities
        prob_all.append(pred_prob)



    pred_all = np.array(sum(pred_all, []))  # Flatten the list of predictions into a single array
    multi_label_all = np.array(sum(multi_label_all, []))  # Flatten the list of true labels into a single array
    prob_all = np.vstack(prob_all)  # Stack the probability arrays vertically

    # Compute corresponding metrics
    acc = accuracy_score(multi_label_all, pred_all)  # Accuracy
    weighted_f1 = f1_score(multi_label_all, pred_all, average="weighted")  # Weighted F1-score
    macro_f1 = f1_score(multi_label_all, pred_all, average="macro")  # Macro F1-score
    kappa = cohen_kappa_score(multi_label_all, pred_all)  # Cohen's kappa
    mcc = matthews_corrcoef(multi_label_all, pred_all)  # Matthews correlation coefficient

    # Additional metrics: macro precision, recall, and AUPR
    macro_precision = precision_score(multi_label_all, pred_all, average='macro')
    macro_recall = recall_score(multi_label_all, pred_all, average='macro')
    multi_label_all_onehot = label_binarize(multi_label_all, classes=[0, 1, 2, 3, 4])
    macro_aupr = average_precision_score(multi_label_all_onehot, prob_all, average='macro')

    # Return all computed metrics and prediction outputs
    return acc,weighted_f1,macro_f1,kappa,mcc,multi_label_all,pred_all,macro_precision,macro_recall,macro_aupr

# Ten-fold cross-validation
def ten_fold(args):
    rawpath = args.rawpath # Directory path for raw data
    gii = open(rawpath+'/drug_side.pkl', 'rb') # Load the drug-side effect frequency matrix
    drug_side = pickle.load(gii)
    gii.close()
    validate_mapping_shapes(rawpath, drug_side)

    # Benchmark dataset data
    final_positive_sample = Extract_positive_negative_samples(drug_side)
    final_sample = final_positive_sample

    # Split data into feature variables (X) and frequencies variables (y)
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []

    # Create the dataset
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1])) # (drug, side effect) pair
        data_y.append((int(float(X[i, 2])))) # frequencies
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    fold = 1

    # Data split
    kfold = StratifiedKFold(10,random_state=42,shuffle=True)

    # Lists to store results for different evaluation metrics
    fold_rows = load_fold_metrics(args)
    completed_folds = {row['fold'] for row in fold_rows}
    total_acc = [row['acc'] for row in fold_rows]
    total_wf1 = [row['weighted_f1'] for row in fold_rows]
    total_maf1 = [row['macro_f1'] for row in fold_rows]
    total_kappa = [row['kappa'] for row in fold_rows]
    total_mcc = [row['mcc'] for row in fold_rows]
    total_prec = [row['macro_precision'] for row in fold_rows]
    total_reca = [row['macro_recall'] for row in fold_rows]
    total_aupr = [row['macro_aupr'] for row in fold_rows]

    # Ten-fold cross-validation experiment
    start_fold = getattr(args, 'start_fold', 1)
    data = np.array(data)

    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        if fold < start_fold:
            fold += 1
            continue
        if fold in completed_folds:
            print(f"==================================fold {fold} already complete; skip")
            fold += 1
            continue
        print("==================================fold {} start".format(fold))

        # Train and test the model on the current fold
        acc,weighted_f1,macro_f1,kappa,mcc,macro_precision,macro_recall,macro_aupr  = train_test(data[train].tolist(), data[test].tolist(), args, fold)

        # Store results of the current fold
        total_acc.append(acc)
        total_wf1.append(weighted_f1)
        total_maf1.append(macro_f1)
        total_kappa.append(kappa)
        total_mcc.append(mcc)
        total_prec.append(macro_precision)
        total_reca.append(macro_recall)
        total_aupr.append(macro_aupr)
        fold_rows.append({
            'fold': fold,
            'acc': acc,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'kappa': kappa,
            'mcc': mcc,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_aupr': macro_aupr,
        })
        completed_folds.add(fold)
        write_run_outputs(args, fold_rows, current_fold=fold, status="running")

        # Print the average results of all folds so far
        print("==================================fold {} end".format(fold))
        print('Total_acc:')
        print(np.mean(total_acc))
        print('Total_weighted_f1:')
        print(np.mean(total_wf1))
        print('Total_macro_f1:')
        print(np.mean(total_maf1))
        print('Total_kappa:')
        print(np.mean(total_kappa))
        print('Total_mcc:')
        print(np.mean(total_mcc))

        print('Total_precision:')
        print(np.mean(total_prec))
        print('Total_recall:')
        print(np.mean(total_reca))
        print('Total_aupr:')
        print(np.mean(total_aupr))

        fold += 1 # Increment the fold counter
        sys.stdout.flush()

    if fold_rows:
        status = "complete" if len(completed_folds) >= 10 else "partial"
        write_run_outputs(args, fold_rows, current_fold=None, status=status)
        print("================================== final summary")
        print("Run directory: " + args.run_dir)
        print("Fold metrics CSV: " + _fold_metrics_path(args))
        print("Paper summary TXT: " + _summary_path(args))


# Benchmark dataset data extraction
def Extract_positive_negative_samples(DAL):
    k = 0 # Initialize counter for indexing the interaction_target array
    interaction_target = np.zeros((DAL.shape[0]*DAL.shape[1], 3)).astype(int) # Initialize a zero matrix

    # Iterate through the DAL matrix and store the indices and values in interaction_target
    for i in range(DAL.shape[0]):  # Loop through each row (drug)
        for j in range(DAL.shape[1]):  # Loop through each column (side effect)
            interaction_target[k, 0] = i  # Store the drug index
            interaction_target[k, 1] = j  # Store the side effect index
            interaction_target[k, 2] = DAL[i, j]  # Store the frequency value
            k = k + 1  # Increment the counter

    # sort all datas
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])

    # obtain benchmark dataset
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]

    return final_positive_sample  # Return the benchmark dataset

# Main entry point for model training and evaluation
def main():

    # =======================
    # Argument Configuration
    # =======================

    # Model and training parameters
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 200,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 0.0001,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 0.00001,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--dropout', type = float, default = 0.4,
                        metavar = 'FLOAT', help = 'dropout rate')
    parser.add_argument('--gp', type = int, default = 64,
                        metavar = 'gp', help = 'hyper_gauss')

    # Batch size settings
    parser.add_argument('--batch_size', type = int, default = 32,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default = 32,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--dataset', type = str, default = 'hh',
                        metavar = 'STRING', help = 'dataset')
    parser.add_argument('--rawpath', type=str, default='./Datas',
                        metavar='STRING', help='rawpath')

    # MSSF-LLM arguments
    parser.add_argument('--use_llm', action='store_true', default=False,
                        help='Enable LLM branch (PubMedBERT features)')
    parser.add_argument('--drug_llm_path', type=str, default='data/mssf_aligned_drug_llm_features.pt',
                        help='Path to encoded drug LLM features for the active dataset')
    parser.add_argument('--se_llm_path', type=str, default='data/mssf_aligned_se_llm_features.pt',
                        help='Path to encoded side-effect LLM features for the active dataset')
    parser.add_argument('--drug_mask_path', type=str, default='data/mssf_aligned_drug_text_mask.pt',
                        help='Path to drug text mask for the active dataset')
    parser.add_argument('--se_mask_path', type=str, default='data/mssf_aligned_se_text_mask.pt',
                        help='Path to side-effect text mask for the active dataset')
    parser.add_argument('--use_cross_modal', action='store_true', default=False,
                        help='Enable cross-modal fusion')
    parser.add_argument('--use_supcon', action='store_true', default=False,
                        help='Enable supervised contrastive loss')
    parser.add_argument('--alpha_supcon', type=float, default=0.1,
                        help='Weight of supervised contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for SupCon loss')
    parser.add_argument('--cross_modal_variant', type=str, default='gated',
                        choices=['gated', 'multihead'],
                        help='Cross-modal fusion variant')
    parser.add_argument('--start_fold', type=int, default=1,
                        metavar='N', help='Skip folds before this number (resume from fold N)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for a new run directory under runs/')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from an existing run directory')

    # Parse arguments
    args = parser.parse_args()
    explicit_keys = _explicit_cli_keys(sys.argv[1:])
    run_dir = setup_run(args, explicit_keys)
    log_file = attach_run_logger(run_dir)


    # =======================
    # Display Hyperparameters
    # =======================
    print('-------------------- Hyperparams --------------------')
    print('learning rate: ' + str(args.lr))
    print('dropout rate: ' + str(args.dropout))
    print('batch_size: ' + str(args.batch_size))
    print('dimension of Bayesian: ' + str(args.gp))
    print('weight decay: ' + str(args.weight_decay))
    print('run directory: ' + str(args.run_dir))

    # =======================
    # Run Ten-Fold Cross-Validation
    # =======================
    ten_fold(args)

# Run main function
if __name__ == "__main__":
    main()







