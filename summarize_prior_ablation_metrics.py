import os
import json
import pandas as pd


EXPERIMENT_NAMES = {
    "raw1": "Baseline 1 (Raw 1-lead)",
    "recon12": "Baseline 2 (Reconstructed 12-lead)",
    "recon12_meta": "Recon 12-lead + Meta",
    "raw1_recon12": "Baseline 3 (Raw + Recon)",
    "raw1_recon12_meta": "Ours (Raw + Recon + Meta)",
    "real12": "Upper Bound (Real 12-lead)",
    "real12_meta": "Upper Bound + Meta",
}


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    base_dir = "checkpoint_ptbxl/prior_ablation"

    rows = []

    for exp, display_name in EXPERIMENT_NAMES.items():
        metrics_path = os.path.join(base_dir, exp, "test_metrics.json")

        if not os.path.exists(metrics_path):
            print("[SKIP] not found:", metrics_path)
            continue

        m = load_metrics(metrics_path)

        rows.append({
            "Method": display_name,
            "Accuracy": m["test_accuracy"],
            "F1-Score": m["test_macro_f1"],
            "AUROC": m["test_macro_auc"],
            "Sensitivity": m["test_sensitivity"],
            "Specificity": m["test_specificity"],
            "Best Epoch": m["best_epoch"],
        })

    df = pd.DataFrame(rows)

    # 보기 좋게 반올림
    show_df = df.copy()
    for col in ["Accuracy", "F1-Score", "AUROC", "Sensitivity", "Specificity"]:
        show_df[col] = show_df[col].map(lambda x: round(x, 4))

    print("\n=== Quantitative Results ===")
    print(show_df.to_string(index=False))

    save_path = os.path.join(base_dir, "quantitative_results_summary.csv")
    show_df.to_csv(save_path, index=False)

    print("\n[SAVE]", save_path)


if __name__ == "__main__":
    main()