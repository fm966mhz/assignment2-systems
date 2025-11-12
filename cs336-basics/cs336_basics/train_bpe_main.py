"""Script to train a BPE tokenizer and save the vocabulary and merges."""

import pickle

from absl import app
from absl import flags
from absl import logging
from cs336_basics.bpe_tokenizer import BpeTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Path to the input text file.")
flags.DEFINE_integer("vocab_size", 1000, "Size of the vocabulary to learn.")
flags.DEFINE_list(
    "special_tokens",
    [],
    "List of special tokens to include in the vocabulary.",
)
flags.DEFINE_string(
    "split_special_token",
    None,
    "Special token used to split the input during pretokenization.",
)
flags.DEFINE_integer(
    "pretokenization_num_processes",
    4,
    "Number of processes to use for pretokenization.",
)
flags.DEFINE_string(
    "output_vocab_path",
    None,
    "Path to save the learned vocabulary (pickled dump).",
)
flags.DEFINE_string(
    "output_merges_path",
    None,
    "Path to save the learned merges (pickled dump).",
)


def main(argv):
    """Main function to train the BPE tokenizer and save vocab and merges."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.input_path is None:
        raise app.UsageError("Must specify --input_path")

    if FLAGS.output_vocab_path is None:
        raise app.UsageError("Must specify --output_vocab_path")

    if FLAGS.output_merges_path is None:
        raise app.UsageError("Must specify --output_merges_path")

    logging.info("Training BPE tokenizer...")
    tokenizer = BpeTokenizer(
        vocab_size=FLAGS.vocab_size,
        special_tokens=FLAGS.special_tokens,
        split_special_token=FLAGS.split_special_token,
        pretokenization_num_processes=FLAGS.pretokenization_num_processes,
    )
    vocab, merges = tokenizer.train(FLAGS.input_path, show_progress=True)

    # Save the learned vocabulary
    with open(FLAGS.output_vocab_path, "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)

    # Save the learned merges
    with open(FLAGS.output_merges_path, "wb") as merges_file:
        pickle.dump(merges, merges_file)

    logging.info("BPE tokenizer training complete.")

    # Find the longest token in the learned vocabulary.
    longest_token = max(vocab.values(), key=len)
    logging.info(
        "Longest token in vocab: %s (length %d)", longest_token, len(longest_token)
    )


if __name__ == "__main__":
    app.run(main)
