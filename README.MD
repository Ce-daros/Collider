# Remove Similar Vectors

This Python script is designed to remove similar vectors from input data using embeddings and DBSCAN clustering. It utilizes the Hugging Face Transformers library for generating embeddings and the scikit-learn library for clustering.

## Install

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/remove-similar-vectors.git
cd remove-similar-vectors
```

2. **Install the required dependencies:**

```python
pip install -r requirements.txt
```

## Usage

Prepare your input data: The script expects a JSON file containing a list of dictionaries, where each dictionary represents an entry with the following keys:

`"system"` (optional): A string representing the system prompt.
`"conversations"`: A list of dictionaries, where each dictionary has a "value" key representing the conversation text.


**Run the script:**

```
python test.py --input path/to/input.json --output path/to/output.json [--model_path path/to/model_directory] [--cutoff_percent PERCENTILE] [--threshold SIMILARITY_THRESHOLD] [--batch_size BATCH_SIZE] [--use_devices DEVICE_IDS] [--min_samples MIN_SAMPLES]
```

`--input:` Path to the input JSON file (default: input.json).

`--output:` Path to the output JSON file (default: input_out.json).

`--model_path:` Path to the directory containing the pre-trained language model (default: /root/autodl-tmp/bge-m3-hf).

`--cutoff_percent:` Percentile for determining the cutoff length for input texts
(default: 95).
`--threshold:` Threshold for similarity score (default: 0.5).

`--batch_size:` Batch size for embedding calculation (default: 256).

`--min_samples:` Minimum number of samples for a core point in DBSCAN (default: 20).

Output: The script will generate an output JSON file containing the unique entries after removing similar vectors based on the specified threshold.

## Feedback

If you have any feedback, suggestions, or issues, please open an issue on the GitHub repository.

## Join our Discord

[Here!](https://discord.gg/CynvTyWe)