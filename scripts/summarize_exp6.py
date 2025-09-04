import os
import pandas as pd


def main():
    project_root = "/home/parkhomenko/Documents/new_project"
    metadata_path = os.path.join(project_root, "data/metadata/metadata.csv")
    summary_out_path = os.path.join(project_root, "data/metadata/exp6_summary.csv")

    classes = [
        "AKIEC",
        "BCC",
        "BKL",
        "DF",
        "MEL",
        "NV",
        "SCC",
        "VASC",
        "NOT_SKIN",
        "UNKNOWN",
    ]

    df = pd.read_csv(metadata_path)
    subset = df[df["exp6"].isin(["train", "val", "test"])][["unified_diagnosis", "exp6"]]
    summary = (
        subset.groupby(["unified_diagnosis", "exp6"]).size().unstack(fill_value=0).reindex(index=classes)
    )
    summary["total"] = summary.sum(axis=1)
    summary.loc["TOTAL"] = summary.sum(axis=0)

    print("Exp6 split sizes:")
    print({k: int(v) for k, v in df["exp6"].value_counts().to_dict().items() if k in ["train", "val", "test"]})
    print("\nPer-class counts:")
    print(summary)

    os.makedirs(os.path.dirname(summary_out_path), exist_ok=True)
    summary.to_csv(summary_out_path)


if __name__ == "__main__":
    main()




