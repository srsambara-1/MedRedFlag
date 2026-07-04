import pandas as pd
import re
import os
import argparse

TEXT_COLUMNS = ["Title", "Body", "Response"]

IMAGE_PATTERN = re.compile(
    r"""(
        attach(?:ed|ment|ing)? |
        photo(?:s|graph|graphic)? |
        pictur(?:e|es|ed|ing)? |
        screenshot(?:s)? |
        \bpic(?:s)?\b |
        image(?:s)? |
        img(?:ur)? |
        jpeg | jpg | png | gif | bmp | tiff | webp | svg |
        upload(?:ed|ing)? |
        \blink(?:ed)?\b |
        \bview(?:\s?this)?\b |
        \bsee\s?(below|above|attached)\b
    )""",
    re.IGNORECASE | re.VERBOSE
)

EDIT_PATTERN = re.compile(
    r"""
    \bedit(?:\s*\d+)?\s*[:.*-]
    """,
    re.IGNORECASE | re.VERBOSE
)

UPDATE_PATTERN = re.compile(r"\bupdate\b", re.IGNORECASE)

PRIVACY_PATTERN = re.compile(
    r"""(
        \bhipaa\b |
        \bphi\b |
        protected\s+health\s+information |
        patient\s+privacy |
        medical\s+privacy |
        privacy\s+concern(?:s)? |
        privacy\s+issue(?:s)? |
        confidentiality |
        confidential(?:ity)? |
        \bconsent\b |
        \bde[-\s]?identified\b |
        \banonym(?:ous|ized|ity)\b |
        \bprivate\s+(?:info|information|details)\b |
        \bwithout\s+(?:my|their)\s+permission\b |
        \bkept\s+private\b |
        \blegal\s+obligation\b
    )""",
    re.IGNORECASE | re.VERBOSE
)



def word_count(text):
    return len(str(text).split())

def has_pattern(row, pattern, columns):
    return any(bool(pattern.search(str(row[col]))) for col in columns)

def too_short(row, min_words=10):
    tb_words = word_count(f"{row['Title']} {row['Body']}")
    resp_words = word_count(row["Response"])
    return tb_words < min_words or resp_words < min_words


def main():
    parser = argparse.ArgumentParser(description="Prefilter MedRedQA datasets")
    parser.add_argument("--input_dir", required=True, help="Input directory containing medredqa_train.csv, medredqa_val.csv, and medredqa_test.csv")
    parser.add_argument("--output", required=True, help="Output CSV path for filtered results")
    args = parser.parse_args()

    input_paths = {
        "train": os.path.join(args.input_dir, "medredqa_train.csv"),
        "val":   os.path.join(args.input_dir, "medredqa_val.csv"),
        "test":  os.path.join(args.input_dir, "medredqa_test.csv"),
    }

    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir or ".", exist_ok=True)

    all_kept = []
    all_removed = []

    global_counts = {
        "total": 0,
        "image": 0,
        "edit": 0,
        "update": 0,
        "privacy": 0,
        "short": 0,
    }

    for split, path in input_paths.items():
        print(f"\n Processing {split}")
        df = pd.read_csv(path)

        # Rename unnamed first column to postID if present
        if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
            df.rename(columns={df.columns[0]: 'postID'}, inplace=True)
        global_counts["total"] += len(df)

        df["has_image"] = df.apply(
            lambda r: has_pattern(r, IMAGE_PATTERN, TEXT_COLUMNS), axis=1
        )
        df["has_edit"] = df.apply(
            lambda r: has_pattern(r, EDIT_PATTERN, TEXT_COLUMNS), axis=1
        )
        df["has_update"] = df.apply(
            lambda r: has_pattern(r, UPDATE_PATTERN, ["Title", "Body"]), axis=1
        )
        df["has_privacy"] = df.apply(
            lambda r: has_pattern(r, PRIVACY_PATTERN, TEXT_COLUMNS), axis=1
        )
        df["too_short"] = df.apply(too_short, axis=1)

        for k, col in [
            ("image", "has_image"),
            ("edit", "has_edit"),
            ("update", "has_update"),
            ("privacy", "has_privacy"),
            ("short", "too_short"),
        ]:
            global_counts[k] += df[col].sum()

        removed = df[
            df["has_image"]
            | df["has_edit"]
            | df["has_update"]
            | df["has_privacy"]
            | df["too_short"]
        ].copy()

        kept = df[
            ~(df["has_image"]
            | df["has_edit"]
            | df["has_update"]
            | df["has_privacy"]
            | df["too_short"])
        ].copy()

        print(f"  Original: {len(df)}")
        print(f"  Kept: {len(kept)}")
        print(f"  Filtered out: {len(removed)}")

        all_kept.append(kept)
        all_removed.append(removed)

    df_kept = pd.concat(all_kept, ignore_index=True)
    df_removed = pd.concat(all_removed, ignore_index=True)

    # Create transformed columns for kept data
    def format_patient_question(row):
        title = str(row.get('Title', '')).strip()
        body = str(row.get('Body', '')).strip()
        if title and body:
            return f"{title}\n\n{body}"
        return title or body

    df_kept['patient_question'] = df_kept.apply(format_patient_question, axis=1)
    df_kept['physician_response'] = df_kept['Response'].astype(str).str.strip()

    # # Save full filtered versions (with all columns)
    # kept_path = os.path.join(args.output_dir, "medredqa_filtered_kept.csv")
    # removed_path = os.path.join(args.output_dir, "medredqa_filtered_out.csv")

    # df_kept.to_csv(kept_path, index=False)
    # df_removed.to_csv(removed_path, index=False)

    # Save transformed version with only the 3 required columns
    transformed_path = args.output
    df_transformed = df_kept[['postID', 'patient_question', 'physician_response']].copy()
    df_transformed.to_csv(transformed_path, index=False)

    print("\n================ SUMMARY ================")
    print(f"Total original rows: {global_counts['total']}")
    print(f"Kept rows: {len(df_kept)}")
    print(f"Filtered rows: {len(df_removed)}")
    print("\nBreakdown (non-exclusive):")
    print(f"  Image references: {global_counts['image']}")
    print(f"  Edit markers: {global_counts['edit']}")
    print(f"  Update markers: {global_counts['update']}")
    print(f"  Privacy markers: {global_counts['privacy']}")
    print(f"  Too short: {global_counts['short']}")
    print("========================================\n")

    # print(f"Saved kept -> {kept_path}")
    # print(f"Saved filtered -> {removed_path}")
    print(f"Saved transformed -> {transformed_path}")


if __name__ == "__main__":
    main()
