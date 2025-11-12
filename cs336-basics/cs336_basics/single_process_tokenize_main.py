"""Tokenize an input text file using a BPE tokenizer and save the token IDs."""

import numpy as np

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

from cs336_basics.bpe_tokenizer import BpeTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string("input_file_path", None, "Path to the input text file to tokenize.")
flags.DEFINE_string(
    "output_token_ids_path", None, "Path to save the token IDs (numpy .npy file)."
)
flags.DEFINE_string(
    "vocab_path", None, "Path to the pickled vocabulary file for the tokenizer."
)
flags.DEFINE_string(
    "merges_path", None, "Path to the pickled merges file for the tokenizer."
)


def main(argv):
    """Main function to tokenize the input file and save token IDs."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.input_file_path is None:
        raise app.UsageError("Must specify --input_file_path")

    if FLAGS.output_token_ids_path is None:
        raise app.UsageError("Must specify --output_token_ids_path")

    if FLAGS.vocab_path is None:
        raise app.UsageError("Must specify --vocab_path")

    if FLAGS.merges_path is None:
        raise app.UsageError("Must specify --merges_path")

    logging.info("Loading BPE tokenizer...")
    tokenizer = BpeTokenizer.from_files(
        vocab_filepath=FLAGS.vocab_path,
        merges_filepath=FLAGS.merges_path,
    )

    logging.info(f"Tokenizing {FLAGS.input_file_path}...")
    with open(FLAGS.input_file_path, "r", encoding="utf-8") as input_file:
        output_token_ids = []
        for token_id in tqdm(tokenizer.encode_iterable(input_file)):
            output_token_ids.append(token_id)
            pass

    logging.info(f"Writing output to {FLAGS.output_token_ids_path}...")
    np.save(FLAGS.output_token_ids_path, np.array(output_token_ids, dtype=np.uint16))
    logging.info("Tokenization complete.")


if __name__ == "__main__":
    app.run(main)
