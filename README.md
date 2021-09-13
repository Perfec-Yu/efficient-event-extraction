# Zero-shot IE

# Generate Weakly Supervised Training Data
`python generate_weakly_supervised_data.py --"proper-args"` arguments explained below. There is a newer version with minor updates (different from the one used in paper) `generate_weakly_supervised_data_v2.py` with the same set of arguments.
- Prepare a corpus of unlabeled sentences `--train-file`.
- Prepare a preprocess function that takes `--train-file` as input argument and returns a list of numerical sentence_ids and sentence strings. (can be stored in seperate python scripts and pass the path as arguments, e.g., `--preprocess-func myutils.utils.preprocess`). There is a default preprocessing that accepts jsonl files (each line being a json string), with each entry having a 'sentence' key for the sentence string. The sentence_id is automatically assigned as the line number.
- Prepare label keywords as in the `data/label_info.json`, and pass as the `--label-json` argument.
- (optional but for better performance) Prepare a few annotated example sentences in `--example-json` with the similar format as `data/example_json`.
- `--encoding-save-dir`, `--corpus-jsonl` are directories and path to save some intermediate outputs.
- `--output-save-dir` is the directory to save the generated outputs.
- `--threshold` is the threshold for annotation (refer to paper). Usually `0.65-0.75` works if you don't have enough example data to decide the value.
- `--evaluate` (`action='store_true'`) instead of use `--threshold`, we can use this to find the best threshold on the `example-json`

# Run Training

`python run_train.py --"proper-args"` arguments explained below.
