import os
import json
import pandas as pd


EXPERIMENT_NAMES = {
    "recon12_meta_concat": "Recon 12-lead + Meta / Concat",
    "recon12_meta_weighted": "Recon 12-lead + Meta / Weighted",
    "recon12_meta_gated": "Recon 12-lead + Meta / Gated",

    "raw1_recon12_meta_concat": "Raw + Recon + Meta / Concat",
    "raw1_recon12_meta_weighted": "Raw + Recon + Meta / Weighted",
    "raw1_recon12_meta_gated": "Raw + Recon + Meta / Gated",
}


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    base_dir = "checkpoint_ptbxl/prior_fusion_epoch30"

    rows = []

    for exp, display_name in EXPERIMENT_NAMES.items():
        metrics_path = os.path.join(base_dir, exp, "test_metrics.json")

        if not os.path.exists(metrics_path):
            print("[SKIP] not found:", metrics_path)
            continue

        m = load_metrics(metrics_path)

        rows.append({
            "Experiment": exp,
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
        print("\n[ERROR] No fusion test_metrics.json files found.")
        print("Check base_dir:", base_dir)
        return

    show_df = df.copy()
    for col in ["Accuracy", "F1-Score", "AUROC", "Sensitivity", "Specificity"]:
        show_df[col] = show_df[col].map(lambda x: round(float(x), 4))

    print("\n=== Fusion Results ===")
    print(show_df.to_string(index=False))

    save_path = os.path.join(base_dir, "fusion_results_summary.csv")
    show_df.to_csv(save_path, index=False)

    print("\n[SAVE]", save_path)

    metric_cols = ["Accuracy", "F1-Score", "AUROC", "Sensitivity", "Specificity"]

    print("\n=== Best Fusion Method by Metric ===")
    for col in metric_cols:
        best_idx = show_df[col].idxmax()
        best_row = show_df.loc[best_idx]
        print(f"{col}: {best_row['Method']} ({best_row[col]:.4f})")


if __name__ == "__main__":
    main()