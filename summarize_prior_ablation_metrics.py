import os
import json
import pandas as pd


BASE_DIR = "checkpoint_ptbxl/prior_ablation_epoch30"

# 왼쪽: 화면에 보여줄 실험 이름
# 오른쪽: 실제로 찾을 수 있는 폴더 후보들
EXPERIMENTS = [
    {
        "display": "Baseline 1 (Raw 1-lead)",
        "candidates": ["raw1"],
    },
    {
        "display": "Raw 1-lead + Meta",
        "candidates": ["raw1_meta", "raw1_meta_concat"],
    },
    {
        "display": "Baseline 2 (Reconstructed 12-lead)",
        "candidates": ["recon12"],
    },
    {
        "display": "Recon 12-lead + Meta",
        "candidates": ["recon12_meta", "recon12_meta_concat"],
    },
    {
        "display": "Baseline 3 (Raw + Recon)",
        "candidates": ["raw1_recon12", "raw1_recon12_concat"],
    },
    {
        "display": "Ours (Raw + Recon + Meta)",
        "candidates": ["raw1_recon12_meta", "raw1_recon12_meta_concat"],
    },
    {
        "display": "Upper Bound (Real 12-lead)",
        "candidates": ["real12"],
    },
    {
        "display": "Upper Bound + Meta",
        "candidates": ["real12_meta", "real12_meta_concat"],
    },
]


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def find_metrics_path(base_dir, candidates):
    for exp in candidates:
        metrics_path = os.path.join(base_dir, exp, "test_metrics.json")
        if os.path.exists(metrics_path):
            return exp, metrics_path

    return None, None


def main():
    rows = []

    for item in EXPERIMENTS:
        display_name = item["display"]
        candidates = item["candidates"]

        used_exp, metrics_path = find_metrics_path(BASE_DIR, candidates)

        if metrics_path is None:
            print("[SKIP] not found:", candidates)
            continue

        m = load_metrics(metrics_path)

        rows.append({
            "Experiment": used_exp,
            "Method": display_name,
            "Accuracy": m["test_accuracy"],
            "F1-Score": m["test_macro_f1"],
            "AUROC": m["test_macro_auc"],
            "Sensitivity": m["test_sensitivity"],
            "Specificity": m["test_specificity"],
            "Best Epoch": m["best_epoch"],
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n[ERROR] No test_metrics.json files found.")
        print("Check base_dir:", BASE_DIR)
        return

    show_df = df.copy()

    for col in ["Accuracy", "F1-Score", "AUROC", "Sensitivity", "Specificity"]:
        show_df[col] = show_df[col].map(lambda x: round(float(x), 4))

    print("\n=== Quantitative Results ===")
    print(show_df.to_string(index=False))

    save_path = os.path.join(BASE_DIR, "quantitative_results_summary.csv")
    show_df.to_csv(save_path, index=False)

    print("\n[SAVE]", save_path)

    metric_cols = ["Accuracy", "F1-Score", "AUROC", "Sensitivity", "Specificity"]

    print("\n=== Best Method by Metric ===")
    for col in metric_cols:
        best_idx = show_df[col].idxmax()
        best_row = show_df.loc[best_idx]
        print(f"{col}: {best_row['Method']} ({best_row[col]:.4f})")


if __name__ == "__main__":
    main()