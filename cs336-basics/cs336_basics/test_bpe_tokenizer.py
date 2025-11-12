import pytest

from cs336_basics.bpe_tokenizer import BpeTokenizer


def test_bpe_pretokenization_single_process():
    """Single process pretokenization test."""
    tokenizer = BpeTokenizer(
        vocab_size=100,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=1,
    )
    pretoken_counts = tokenizer.pretokenize(
        "cs336_basics/test_data/pretokenizer_test.txt"
    )
    assert isinstance(pretoken_counts, dict)

    expected_pretoken_counts = {
        b" ": 3,
        b"test": 2,
        b"\n": 2,
        b"hello": 1,
        b",": 1,
        b" another": 1,
        b" test": 1,
        b" \xe6\xb5\x8b\xe8\xaf\x95": 1,
        b" \xe4\xb8\xad\xe6\x96\x87": 1,
        b"\xe5\x8f\xa6\xe5\xa4\x96\xe4\xb8\x80\xe4\xb8\xaa": 1,
    }
    assert pretoken_counts == expected_pretoken_counts


def test_bpe_pretokenization_multi_process():
    """Multi-process pretokenization test."""
    tokenizer = BpeTokenizer(
        vocab_size=100,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=4,
    )
    pretoken_counts = tokenizer.pretokenize(
        "cs336_basics/test_data/pretokenizer_test.txt"
    )
    assert isinstance(pretoken_counts, dict)

    expected_pretoken_counts = {
        b" ": 3,
        b"test": 2,
        b"\n": 2,
        b"hello": 1,
        b",": 1,
        b" another": 1,
        b" test": 1,
        b" \xe6\xb5\x8b\xe8\xaf\x95": 1,
        b" \xe4\xb8\xad\xe6\x96\x87": 1,
        b"\xe5\x8f\xa6\xe5\xa4\x96\xe4\xb8\x80\xe4\xb8\xaa": 1,
    }
    assert pretoken_counts == expected_pretoken_counts


def test_tokenizer_train_single_process():
    """Test training the BPE tokenizer using a single process."""
    tokenizer = BpeTokenizer(
        vocab_size=(2 + 256 + 6),
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        pretokenization_num_processes=1,
    )
    vocab, merges = tokenizer.train("cs336_basics/test_data/tokenizer_train_test.txt")
    expected_merges = [
        (b"s", b"t"),
        (b"e", b"st"),
        (b"o", b"w"),
        (b"l", b"ow"),
        (b"w", b"est"),
        (b"n", b"e"),
    ]

    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    assert len(vocab) == (2 + 256 + 6)  # special tokens + byte vocab + 6 merges
    assert all(token in vocab.values() for token in [b"<|endoftext|>", b"<|pad|>"])
    assert merges == expected_merges


def test_encode_one_pretoken():
    """Test encoding a single pretoken."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"t",
            3: b"h",
            4: b"e",
            5: b"th",
            6: b"he",
            7: b"the",
        },
        merges=[(b"h", b"e"), (b"t", b"h"), (b"th", b"e"), (b"t", b"he")],
    )

    token_ids = tokenizer.encode("the")
    expected_token_ids = [7]  # "the" is a single token in the vocab
    assert token_ids == expected_token_ids


def test_encode_one_pretoken_repeated_bytes():
    """Test encoding a single pretoken."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"l",
            3: b"ll",
            4: b"lll",
            5: b"llll",
        },
        merges=[(b"l", b"l"), (b"ll", b"l"), (b"lll", b"l"), (b"l", b"ll")],
    )
    assert tokenizer.encode("llll") == [3, 3]

    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"l",
            3: b"ll",
            4: b"lll",
            5: b"llll",
        },
        merges=[
            (b"l", b"l"),
            (b"ll", b"l"),
            (b"lll", b"l"),
            (b"l", b"ll"),
            (b"ll", b"ll"),
        ],
    )
    assert tokenizer.encode("llll") == [5]  # "llll" is a single token in the vocab


def test_encode_with_special_tokens():
    """Test encoding text with special tokens."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"h",
            3: b"i",
            4: b"hi",
            5: b"!",
        },
        merges=[(b"h", b"i"), (b"i", b"!")],
    )

    text = "hi<|endoftext|>hi!<|pad|>hi!hi!"
    token_ids = tokenizer.encode(text)
    # "hi", "<|endoftext|>", "hi", "!", "<|pad|>", "hi", "!", "hi", "!"
    expected_token_ids = [4, 0, 4, 5, 1, 4, 5, 4, 5]
    assert token_ids == expected_token_ids


def test_encode_iterable_without_special_tokens():
    """Test encoding an iterable of texts with special tokens."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"h",
            1: b"i",
            2: b"hi",
            3: b"!",
        },
        merges=[(b"h", b"i"), (b"i", b"!")],
    )

    texts = ["hihi", "hi!hi!"]
    token_ids_list = list(tokenizer.encode_iterable(texts))
    expected_token_ids_list = [2, 2, 2, 3, 2, 3]
    assert token_ids_list == expected_token_ids_list


def test_encode_iterable_with_special_tokens():
    """Test encoding an iterable of texts with special tokens."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|abc|>", "<|def|>"],
        split_special_token="<|abc|>",
        vocab={
            0: b"<|abc|>",
            1: b"<|def|>",
            2: b"h",
            3: b"i",
            4: b"hi",
            5: b"!",
        },
        merges=[(b"h", b"i"), (b"i", b"!")],
    )

    texts = ["hi<|ab", "c|>hi", "!<|def|", ">hi", "!hi!"]
    token_ids_list = list(tokenizer.encode_iterable(texts))
    # "hi", "<|abc|>", "hi", "!", "<|def|>", "hi", "!", "hi", "!"
    expected_token_ids_list = [4, 0, 4, 5, 1, 4, 5, 4, 5]
    assert token_ids_list == expected_token_ids_list


def test_roundtrip_empty():
    """Test encoding and decoding an empty string."""
    tokenizer = BpeTokenizer(
        special_tokens=["<|endoftext|>", "<|pad|>"],
        split_special_token="<|endoftext|>",
        vocab={
            0: b"<|endoftext|>",
            1: b"<|pad|>",
            2: b"h",
            3: b"i",
            4: b"hi",
            5: b"!",
        },
        merges=[(b"h", b"i"), (b"i", b"!")],
    )
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# def test_tmp_debug_example():
#     """Temporary debug example to print token IDs at specific indices."""
#     tokenizer = BpeTokenizer.from_files(
#         vocab_filepath=(
#             "/Users/danjiewenren/Research/classes/cs336/notes_and_writeups/assignment1_output/vocab_tiny_stories_train.p"
#         ),
#         merges_filepath=(
#             "/Users/danjiewenren/Research/classes/cs336/notes_and_writeups/assignment1_output/merges_tiny_stories_train.p"
#         ),
#     )
#     with open(
#         "/Users/danjiewenren/Research/classes/cs336/notes_and_writeups/data/TinyStoriesV2-GPT4-valid.txt",
#         "r",
#         encoding="utf-8",
#     ) as f:
#         i = 0
#         for token in tokenizer.encode_iterable(f):
#             if 407746 <= i <= 407758:
#                 print(f"Index {i}: Token ID {token}")
#             if i > 407758:
#                 break
#             i += 1
#     test_chunk = "fun!\n"
#     test_lookahead = "The end.\n<|endoftext|>\n"
#     print(
#         f"test_chunk size: {len(test_chunk)}, test_lookahead size: {len(test_lookahead)}"
#     )
#     tokenizer._init_for_encoding()
#     token_ids, num_chars_tokenized = tokenizer._encode_one_chunk(
#         test_chunk, test_lookahead
#     )
#     print(f"Token IDs: {token_ids}")
#     print(f"Number of characters tokenized: {num_chars_tokenized}")
#     print(f"Token IDs decoded: {tokenizer.decode(token_ids)}")
#     leftover_text = (test_chunk + test_lookahead)[num_chars_tokenized:]
#     print(f"Leftover text after tokenization: [{leftover_text}]")
