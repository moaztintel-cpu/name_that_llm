import os

# Set thread limits before importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_ensemble_v2 as base

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
EPOCHS = 120
BATCH_SIZE = 128
MODEL_COUNT = len(base.MODEL_NAMES)


class MasterNetwork(nn.Module):
    def __init__(self, input_size, class_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, class_count),
        )

    def forward(self, x):
        return self.network(x)


def dataset_fingerprint(train):
    payload = pd.util.hash_pandas_object(
        train[["text", "model", "prompt_id"]], index=True
    ).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def balanced_accuracy_fast(y_true, y_pred, class_count):
    """Same balanced-accuracy calculation used by experiment.py."""
    correct = np.bincount(
        y_true[y_true == y_pred], minlength=class_count
    )
    totals = np.bincount(y_true, minlength=class_count)
    return float(np.mean(correct / np.maximum(totals, 1)))


def gray_code(number):
    return number ^ (number >> 1)


def select_best_subset(inner_oof, y_train, class_count, fold_number):
    """Select among all non-empty predictor subsets using outer-train OOF only.

    This mirrors experiment.py's Gray-code exhaustive search. The outer
    validation data is never passed into this function.
    """
    model_count = inner_oof.shape[1]
    if model_count != MODEL_COUNT:
        raise RuntimeError(
            f"Expected {MODEL_COUNT} predictors, got {model_count}."
        )

    total_combinations = (1 << model_count) - 1
    previous_mask = 0
    running_sum = np.zeros(
        (len(y_train), class_count), dtype=np.float32
    )
    best_score = -1.0
    best_mask = None
    best_indices = None

    print(
        f"\nOuter {fold_number}: selecting from "
        f"{total_combinations:,} subsets using INNER OOF only...",
        flush=True,
    )

    for k in range(1, 1 << model_count):
        mask = gray_code(k)
        changed = mask ^ previous_mask
        bit = changed.bit_length() - 1

        if mask & (1 << bit):
            running_sum += inner_oof[:, bit, :]
        else:
            running_sum -= inner_oof[:, bit, :]

        prediction = running_sum.argmax(axis=1)
        score = balanced_accuracy_fast(
            y_train, prediction, class_count
        )

        # Deterministic tie-breaking: keep the first subset at the maximum.
        if score > best_score + 1e-12:
            best_score = score
            best_mask = mask
            best_indices = [
                i for i in range(model_count) if mask & (1 << i)
            ]

        previous_mask = mask

        if k % 4096 == 0:
            print(
                f"  Checked {k:,}/{total_combinations:,} | "
                f"best inner score so far: {best_score:.5f}",
                flush=True,
            )

    if best_mask is None or not best_indices:
        raise RuntimeError("Exhaustive subset search selected no predictors.")

    names = [base.MODEL_NAMES[i] for i in best_indices]
    print(
        f"Outer {fold_number}: selected {len(best_indices)} predictors | "
        f"inner OOF balanced accuracy={best_score:.5f}",
        flush=True,
    )
    print("  " + " | ".join(names), flush=True)

    return best_mask, np.asarray(best_indices, dtype=np.int64), best_score


def train_meta(oof, y, class_count, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = oof.reshape(len(oof), -1).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = MasterNetwork(MODEL_COUNT * class_count, class_count)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(X_tensor))
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tensor[indices])
            loss = criterion(logits, y_tensor[indices])
            loss.backward()
            optimizer.step()

    model.eval()
    return model, mean, std


def predict_meta(model, probabilities, mean, std):
    X = probabilities.reshape(len(probabilities), -1).astype(np.float32)
    X = (X - mean) / std
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).cpu().numpy()


def save_checkpoint(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main():
    torch.set_num_threads(4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("FAIR NESTED GROUP VALIDATION — MASTER vs FOLD-SELECTED SUBSET", flush=True)
    print("=" * 72, flush=True)
    print(f"Predictors: {MODEL_COUNT}", flush=True)
    print(
        f"Subsets searched inside each outer fold: {(1 << MODEL_COUNT) - 1:,}",
        flush=True,
    )

    train, _ = base.load_data()
    required = {"text", "model", "prompt_id"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    data_hash = dataset_fingerprint(train)
    classes = sorted(train["model"].unique().tolist())
    label_to_id = {name: i for i, name in enumerate(classes)}
    y = np.asarray(
        [label_to_id[value] for value in train["model"]], dtype=np.int64
    )
    texts = train["text"].fillna("").tolist()
    groups = train["prompt_id"].to_numpy()
    class_count = len(classes)

    if len(np.unique(groups)) < OUTER_FOLDS:
        raise ValueError("Not enough distinct prompt groups for outer validation.")

    outer = GroupKFold(n_splits=OUTER_FOLDS)
    fold_rows = []

    for outer_number, (outer_train, outer_valid) in enumerate(
        outer.split(texts, y, groups), start=1
    ):
        checkpoint = RESULTS / f"fair_validation_outer_{outer_number}.npz"

        print("\n" + "=" * 72, flush=True)
        print(f"OUTER FOLD {outer_number}/{OUTER_FOLDS}", flush=True)
        print(
            f"Training: {len(outer_train)} | Validation: {len(outer_valid)}",
            flush=True,
        )
        print("=" * 72, flush=True)

        if checkpoint.exists():
            print("Loading saved fair outer-fold checkpoint.", flush=True)
            with np.load(checkpoint, allow_pickle=False) as saved:
                saved_hash = str(saved["dataset_hash"].item())
                if saved_hash != data_hash:
                    raise RuntimeError(
                        f"Dataset fingerprint mismatch in {checkpoint}. "
                        "Move the fair checkpoint aside before rerunning."
                    )
                if not np.array_equal(saved["valid_idx"], outer_valid):
                    raise RuntimeError(
                        f"Validation indices do not match in {checkpoint}."
                    )
                master_pred = saved["master_pred"].copy()
                average_pred = saved["average_pred"].copy()
                selected_pred = saved["selected_pred"].copy()
                true_y = saved["true_y"].copy()
                selected_indices = saved["selected_indices"].astype(np.int64)
                selected_mask = int(saved["selected_mask"].item())
                selection_score = float(
                    saved["inner_selection_score"].item()
                )

            if not np.array_equal(true_y, y[outer_valid]):
                raise RuntimeError(f"Saved labels do not match in {checkpoint}.")
        else:
            inner = GroupKFold(n_splits=INNER_FOLDS)
            inner_oof = np.full(
                (len(outer_train), MODEL_COUNT, class_count),
                np.nan,
                dtype=np.float32,
            )

            outer_texts = [texts[i] for i in outer_train]
            outer_labels = [train["model"].iloc[i] for i in outer_train]
            outer_groups = groups[outer_train]
            outer_y = np.asarray(
                [label_to_id[value] for value in outer_labels], dtype=np.int64
            )

            for inner_number, (inner_train, inner_valid) in enumerate(
                inner.split(outer_texts, outer_y, outer_groups), start=1
            ):
                print(
                    f"\nOuter {outer_number}: inner fold "
                    f"{inner_number}/{INNER_FOLDS}",
                    flush=True,
                )
                flat = base.fit_all_models(
                    [outer_texts[i] for i in inner_train],
                    [outer_labels[i] for i in inner_train],
                    [outer_texts[i] for i in inner_valid],
                    classes,
                    SEED + outer_number * 100 + inner_number,
                )
                inner_oof[inner_valid] = np.asarray(
                    flat, dtype=np.float32
                ).reshape(len(inner_valid), MODEL_COUNT, class_count)

            if not np.isfinite(inner_oof).all():
                raise RuntimeError("Inner OOF predictions are incomplete.")

            # Both the master and subset selector learn from outer-training
            # OOF data only. Neither sees outer-validation labels.
            selected_mask, selected_indices, selection_score = select_best_subset(
                inner_oof, outer_y, class_count, outer_number
            )

            print(
                "\nGenerating predictions for untouched outer validation prompts...",
                flush=True,
            )
            flat_valid = base.fit_all_models(
                outer_texts,
                outer_labels,
                [texts[i] for i in outer_valid],
                classes,
                SEED + outer_number * 10000,
            )
            valid_probs = np.asarray(
                flat_valid, dtype=np.float32
            ).reshape(len(outer_valid), MODEL_COUNT, class_count)

            model, mean, std = train_meta(
                inner_oof,
                outer_y,
                class_count,
                SEED + outer_number,
            )
            master_pred = predict_meta(model, valid_probs, mean, std)
            average_pred = valid_probs.mean(axis=1).argmax(axis=1)
            selected_pred = (
                valid_probs[:, selected_indices, :].mean(axis=1).argmax(axis=1)
            )
            true_y = y[outer_valid]

            save_checkpoint(
                checkpoint,
                valid_idx=outer_valid,
                master_pred=master_pred,
                average_pred=average_pred,
                selected_pred=selected_pred,
                true_y=true_y,
                selected_indices=selected_indices,
                selected_mask=np.asarray(selected_mask, dtype=np.int64),
                inner_selection_score=np.asarray(
                    selection_score, dtype=np.float64
                ),
                dataset_hash=np.asarray(data_hash),
            )
            print(f"Saved fair outer fold {outer_number}.", flush=True)

        selected_names = [base.MODEL_NAMES[i] for i in selected_indices.tolist()]
        fold_result = {
            "fold": outer_number,
            "rows": len(true_y),
            "master_balanced_accuracy": balanced_accuracy_score(true_y, master_pred),
            "average15_balanced_accuracy": balanced_accuracy_score(true_y, average_pred),
            "fold_selected_balanced_accuracy": balanced_accuracy_score(true_y, selected_pred),
            "master_accuracy": accuracy_score(true_y, master_pred),
            "average15_accuracy": accuracy_score(true_y, average_pred),
            "fold_selected_accuracy": accuracy_score(true_y, selected_pred),
            "inner_selection_balanced_accuracy": selection_score,
            "selected_predictor_count": len(selected_indices),
            "selected_mask": selected_mask,
            "selected_predictors": "|".join(selected_names),
        }
        fold_rows.append(fold_result)

        print("\nFOLD RESULTS", flush=True)
        for key, value in fold_result.items():
            if isinstance(value, (float, np.floating)):
                print(f"{key}: {value:.5f}", flush=True)
            else:
                print(f"{key}: {value}", flush=True)

    # Each row must receive exactly one untouched outer-fold prediction.
    all_indices = []
    all_master = []
    all_average = []
    all_selected = []
    all_true = []

    for fold in range(1, OUTER_FOLDS + 1):
        path = RESULTS / f"fair_validation_outer_{fold}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["dataset_hash"].item()) != data_hash:
                raise RuntimeError(f"Dataset fingerprint mismatch in {path}.")
            all_indices.extend(saved["valid_idx"].tolist())
            all_master.extend(saved["master_pred"].tolist())
            all_average.extend(saved["average_pred"].tolist())
            all_selected.extend(saved["selected_pred"].tolist())
            all_true.extend(saved["true_y"].tolist())

    if len(all_indices) != len(train):
        raise RuntimeError(
            f"Expected {len(train)} outer predictions, got {len(all_indices)}."
        )
    if len(set(all_indices)) != len(train):
        raise RuntimeError("Outer validation indices contain duplicates or gaps.")
    if sorted(all_indices) != list(range(len(train))):
        raise RuntimeError("Outer validation indices do not cover the dataset exactly.")

    order = np.argsort(np.asarray(all_indices))
    true_y = np.asarray(all_true, dtype=np.int64)[order]
    master_pred = np.asarray(all_master, dtype=np.int64)[order]
    average_pred = np.asarray(all_average, dtype=np.int64)[order]
    selected_pred = np.asarray(all_selected, dtype=np.int64)[order]

    summary = pd.DataFrame(
        [
            {
                "model": "PyTorch master network",
                "balanced_accuracy": balanced_accuracy_score(true_y, master_pred),
                "accuracy": accuracy_score(true_y, master_pred),
            },
            {
                "model": "Average of all 15 predictors",
                "balanced_accuracy": balanced_accuracy_score(true_y, average_pred),
                "accuracy": accuracy_score(true_y, average_pred),
            },
            {
                "model": "Per-fold selected subset (selection on outer-training OOF only)",
                "balanced_accuracy": balanced_accuracy_score(true_y, selected_pred),
                "accuracy": accuracy_score(true_y, selected_pred),
            },
        ]
    )

    summary_path = RESULTS / "fair_nested_validation_summary.csv"
    folds_path = RESULTS / "fair_nested_validation_folds.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)

    print("\n" + "=" * 72, flush=True)
    print("FINAL FAIR NESTED VALIDATION RESULTS", flush=True)
    print("=" * 72, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved summary: {summary_path}", flush=True)
    print(f"Saved fold selections: {folds_path}", flush=True)
    print(
        "\nReminder: the selected subset may differ between folds. "
        "The combined score evaluates the selection procedure, not one fixed subset.",
        flush=True,
    )


if __name__ == "__main__":
    main()
