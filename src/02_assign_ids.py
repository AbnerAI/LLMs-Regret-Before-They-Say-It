"""Stage 2 - attach a stable integer ID to every record of the regret dataset.

The ID is the line number in the stage-1 output and becomes the key that ties a
record to its extracted hidden states (`results/regret_<size>/...`) throughout
the rest of the pipeline, so it must be assigned once and never reshuffled.

Usage:
    python src/02_assign_ids.py \
        --input data/regret_dataset.json \
        --output data/regret_dataset_with_id.json
"""

import argparse
import json


def add_id_to_json(input_file_path, output_file_path):
    """Copy a JSON-lines file, prepending an "ID" field to every record."""
    with open(input_file_path, "r", encoding="utf-8") as infile, \
            open(output_file_path, "w", encoding="utf-8") as outfile:
        for id_value, line in enumerate(infile):
            entry = json.loads(line)
            entry = {"ID": id_value, **entry}
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/regret_dataset.json")
    parser.add_argument("--output", default="data/regret_dataset_with_id.json")
    args = parser.parse_args()
    add_id_to_json(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
