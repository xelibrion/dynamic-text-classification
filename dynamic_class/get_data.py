#!/usr/bin/env python
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


SEED = 42


def cut_header(text):
    lines = [x for x in text.split("\n") if x]
    idx = -1
    for idx, l in enumerate((lines)):
        if l.startswith("Lines:"):
            break
    return "\n".join(lines[idx + 1 :])


def main(out_dir: Path):
    train = fetch_20newsgroups(subset="all")
    df_train = pd.DataFrame({"text": train["data"], "target": train["target"]})
    df_train["text"] = df_train["text"].apply(cut_header)
    df_train = df_train[df_train["text"] != ""]

    df_train, df_val = train_test_split(
        df_train, train_size=10000, random_state=SEED, stratify=df_train["target"]
    )
    df_val, df_test = train_test_split(
        df_val, train_size=0.5, random_state=SEED, stratify=df_val["target"]
    )
    print(f"Train dataset: {df_train.shape[0]} records")
    print(f"Val dataset: {df_val.shape[0]} records")
    print(f"Test dataset: {df_test.shape[0]} records")
    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    main(Path(__file__).parent / "input")
