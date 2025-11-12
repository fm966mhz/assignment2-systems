"""Byte Pair Encoding (BPE) tokenizer implementation."""

import copy
import pickle
import heapq

from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Iterator

import regex as re

from memory_profiler import profile
from tqdm import tqdm

from .pretokenization_example import find_chunk_boundaries


_DEFAULT_PRETOKENIZATION_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_SPLIT_SPECIAL_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class BytesPair:
    """A pair of bytes used in BPE merges.

    This is mostly needed because the solution wants to break ties by taking the max of the bytes
    pairs of the same count. `heapq` in Python is a min-heap, so we need to create a custom class
    and override the less-than operator to achieve this.
    """

    first: bytes
    second: bytes

    def __lt__(self, other: "BytesPair") -> bool:
        """Less-than operator for BytesPair.

        We want to break ties by taking the max of the bytes pairs, so we invert the comparison.
        """
        if self.first != other.first:
            return self.first > other.first
        return self.second > other.second


@dataclass
class BytesPairListNode:
    """A node in the bytes pair linked list."""

    bytes_pair: BytesPair
    prev: "BytesPairListNode | None" = None
    next: "BytesPairListNode | None" = None

    def __lt__(self, other: "BytesPairListNode") -> bool:
        return self.bytes_pair < other.bytes_pair

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass
class PretokenInfo:
    """Information about a pretoken in the BPE training process."""

    count: int
    first_node: BytesPairListNode | None = None


class BpeTokenizer:
    """A Byte Pair Encoding tokenizer implementation."""

    def __init__(
        self,
        *,
        vocab_size: int | None = None,
        pretokenization_regex: str = _DEFAULT_PRETOKENIZATION_REGEX,
        split_special_token: str | None = _SPLIT_SPECIAL_TOKEN,
        special_tokens: list[str] | None = None,
        pretokenization_num_processes: int = 4,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        """Initialize the BPE tokenizer.
        Args:
            vocab_size (int): The desired vocabulary size.
            pretokenization_regex (str): The regex pattern for pretokenization.
            split_special_token (str | None): Special token used to split the input during
                pretokenization.
            special_tokens (list[str] | None): List of special tokens to include in the vocabulary.
            pretokenization_num_processes (int): Number of processes to use for pretokenization.
            vocab (dict[int, bytes] | None): Predefined vocabulary to use. If provided, training
                related arguments will be ignored, such as vocab_size.
            merges (list[tuple[bytes, bytes]] | None): Predefined merges to use. If provided,
                training related arguments will be ignored, such as vocab_size.
        """
        # Compile the pretokenization regex pattern.
        self._pretokenization_regex_pattern = re.compile(
            pretokenization_regex.encode("utf-8")
        )
        # Handle special tokens.
        self._split_special_token = (
            split_special_token or _SPLIT_SPECIAL_TOKEN
        ).encode("utf-8")
        self._special_tokens = (
            {special_token.encode("utf-8") for special_token in special_tokens}
            if special_tokens
            else set()
        )
        if self._split_special_token not in self._special_tokens:
            self._special_tokens.add(self._split_special_token)
        self._special_tokens_split_regex = b"|".join(
            # Sorting to ensure that longer special tokens are used for splitting.
            # This tested in `tests/test_tokenizer.py::test_overlapping_special_tokens`.
            sorted(
                [re.escape(token) for token in self._special_tokens],
                key=len,
                reverse=True,
            )
        )
        # Number of processes to use for pretokenization.
        self._pretokenization_num_processes = pretokenization_num_processes
        # Trained vocab and merges will be stored here after training.
        if vocab is not None and merges is not None:
            self._vocab = vocab
            self._merges = [
                BytesPair(first=merge[0], second=merge[1]) for merge in merges
            ]
            self._vocab_size = len(vocab)
        elif vocab is None and merges is None and vocab_size is not None:
            self._vocab: dict[int, bytes] = dict(
                enumerate(sorted(self._special_tokens))
            )
            for i in range(256):
                self._vocab[i + len(self._special_tokens)] = bytes([i])
            self._merges: list[BytesPair] = []
            self._vocab_size = vocab_size
        else:
            raise ValueError(
                "Both vocab and merges must be provided together, or neither."
            )
        # Data structures needed for encoding.
        self._merges_to_idx: dict[BytesPair, int] = {}
        self._inverted_vocab: dict[bytes, int] = {}
        # Longest vocab token length in bytes.
        self._longest_vocab_length: int = 0

    def train(
        self, input_path: str, show_progress: bool = False
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        Train the BPE tokenizer on the given input file.

        Args:
            input_path (str): Path to the input text file.
        """
        pretoken_counts = self.pretokenize(input_path)
        pretoken_info_list, bytes_pair_to_pretoken_positions, bytes_pair_counts = (
            self._preprocess_pretoken_counts(pretoken_counts)
        )
        # Build the initial heap of byte pairs based on their counts.
        heap: list[tuple[int, BytesPair]] = [
            (-count, bytes_pair) for bytes_pair, count in bytes_pair_counts.items()
        ]
        heapq.heapify(heap)
        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=self._vocab_size,
                initial=len(self._vocab),
                desc="Merging BPE pairs",
            )
        # Perform BPE merges until reaching the desired vocab size.
        while len(self._vocab) < self._vocab_size and heap:
            neg_count, most_frequent_pair = heapq.heappop(heap)
            current_count = bytes_pair_counts.get(most_frequent_pair, 0)
            # If the count has changed, drop the pair or reinsert with the updated count.
            if current_count == 0:
                continue
            if -neg_count != current_count:
                heapq.heappush(heap, (-current_count, most_frequent_pair))
                continue
            # Perform the merge.
            new_token = most_frequent_pair.first + most_frequent_pair.second
            self._vocab[len(self._vocab)] = new_token
            self._merges.append(most_frequent_pair)
            if progress_bar is not None:
                progress_bar.update(1)

            # Update pretokens containing the merged pair.
            self._bpe_merge(
                most_frequent_pair,
                bytes_pair_to_pretoken_positions,
                bytes_pair_counts,
                heap,
                pretoken_info_list,
            )
        if progress_bar is not None:
            progress_bar.close()
        return self._vocab, self.get_merge_as_list_of_tuples()

    def _pretokenize_one_chunk(
        self,
        input_path: str,
        start: int,
        end: int,
    ) -> Counter[bytes]:
        """
        Pretokenize the input text into a list of byte strings.

        Args:
            input_path (str): Path to the input text file.
            start (int): Start byte position of the chunk.
            end (int): End byte position of the chunk.

        Returns:
            Counter[bytes]: Counter of byte strings representing the pretokenized text.
        """
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk_size = end - start
            chunk_bytes = f.read(chunk_size)
        pretoken_counts: Counter[bytes] = Counter()
        for split_part in re.split(self._special_tokens_split_regex, chunk_bytes):
            if not split_part:
                continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                pretoken = match.group(0)
                pretoken_counts[pretoken] += 1
        return pretoken_counts

    def pretokenize(
        self,
        input_path: str,
    ) -> Counter[bytes]:
        """
        Pretokenize the input file into a list of byte strings.

        Args:
            input_path (str): Path to the input text file.

        Returns:
            Counter[bytes]: Counter of byte strings representing the pretokenized text.
        """
        pretoken_counts: Counter[bytes] = Counter()

        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=self._pretokenization_num_processes,
                split_special_token=self._split_special_token,
            )

        with Pool(processes=self._pretokenization_num_processes) as pool:
            results = []
            for bi in range(len(chunk_boundaries) - 1):
                start = chunk_boundaries[bi]
                end = chunk_boundaries[bi + 1]
                results.append(
                    pool.apply_async(
                        self._pretokenize_one_chunk,
                        args=(input_path, start, end),
                    )
                )
            for r in results:
                pretoken_counts += r.get()

        return pretoken_counts

    def _preprocess_pretoken_counts(
        self,
        pretoken_counts: Counter[bytes],
    ) -> tuple[
        list[PretokenInfo],
        dict[BytesPair, set[int]],
        dict[BytesPair, int],
    ]:
        """
        Preprocess the pretoken counts into a structure suitable for BPE training.

        Args:
            pretoken_counts (Counter[bytes]): Counter of byte strings representing the pretokenized
                text.

        Returns:
            list[PretokenInfo]: List of PretokenInfo objects for BPE training.
            dict[BytesPair, set[int]]: inverted list from bytes pairs to the positions of the
                pretokens that generate such pairs.
            dict[BytesPair, int]: counts of each bytes pair across all pretokens.
        """
        pretoken_info_list: list[PretokenInfo] = []
        bytes_pair_to_pretoken_positions: dict[BytesPair, set[int]] = defaultdict(set)
        bytes_pair_counts: dict[BytesPair, int] = defaultdict(int)

        for pretoken, count in pretoken_counts.items():
            # Build the linked list of byte pairs for this pretoken
            first_node: BytesPairListNode | None = None
            prev_node: BytesPairListNode | None = None
            byte_sequence = list(pretoken)
            for i in range(len(byte_sequence) - 1):
                bytes_pair = BytesPair(
                    first=bytes([byte_sequence[i]]),
                    second=bytes([byte_sequence[i + 1]]),
                )
                bytes_pair_counts[bytes_pair] += count
                current_node = BytesPairListNode(bytes_pair=bytes_pair)
                if first_node is None:
                    first_node = current_node
                if prev_node is not None:
                    prev_node.next = current_node
                    current_node.prev = prev_node
                prev_node = current_node

                # Update the inverted index
                bytes_pair_to_pretoken_positions[bytes_pair].add(
                    len(pretoken_info_list)
                )

            pretoken_info = PretokenInfo(count=count, first_node=first_node)
            pretoken_info_list.append(pretoken_info)

        return pretoken_info_list, bytes_pair_to_pretoken_positions, bytes_pair_counts

    def _bpe_merge(
        self,
        most_frequent_pair: BytesPair,
        bytes_pair_to_pretoken_positions: dict[BytesPair, set[int]],
        bytes_pair_counts: dict[BytesPair, int],
        search_heap: list[tuple[int, BytesPair]],
        pretoken_info_list: list[PretokenInfo],
    ) -> None:
        # TODO(djwenren): this could be further optimized by recording the nodes in the linked
        # list that contain the most frequent pair, so we don't have to traverse the entire list.
        new_token = most_frequent_pair.first + most_frequent_pair.second
        newly_generated_pair_counts: dict[BytesPair, int] = defaultdict(int)
        for pos in bytes_pair_to_pretoken_positions[most_frequent_pair]:
            pretoken_info = pretoken_info_list[pos]
            node = pretoken_info.first_node
            while node is not None:
                if node.bytes_pair != most_frequent_pair:
                    node = node.next
                    continue
                # Update the previous node's next pointer and bytes pair.
                if node.prev is None:
                    pretoken_info.first_node = node.next
                else:
                    node.prev.next = node.next
                    new_bytes_pair = BytesPair(
                        first=node.prev.bytes_pair.first,
                        second=new_token,
                    )
                    old_bytes_pair = node.prev.bytes_pair
                    node.prev.bytes_pair = new_bytes_pair
                    bytes_pair_counts[old_bytes_pair] -= pretoken_info.count
                    newly_generated_pair_counts[new_bytes_pair] += pretoken_info.count
                    bytes_pair_to_pretoken_positions[new_bytes_pair].add(pos)
                # Update the next node's previous pointer and bytes pair.
                if node.next is not None:
                    node.next.prev = node.prev
                    new_bytes_pair = BytesPair(
                        first=new_token,
                        second=node.next.bytes_pair.second,
                    )
                    old_bytes_pair = node.next.bytes_pair
                    node.next.bytes_pair = new_bytes_pair
                    bytes_pair_counts[old_bytes_pair] -= pretoken_info.count
                    newly_generated_pair_counts[new_bytes_pair] += pretoken_info.count
                    bytes_pair_to_pretoken_positions[new_bytes_pair].add(pos)
                # Move to the next node.
                node = node.next
        # Update counts for newly generated byte pairs.
        for bytes_pair, count in newly_generated_pair_counts.items():
            if count > 0:
                bytes_pair_counts[bytes_pair] += count
                heapq.heappush(
                    search_heap,
                    (
                        -count,
                        bytes_pair,
                    ),
                )
        # Remove the merged pair from counts and positions.
        del bytes_pair_counts[most_frequent_pair]
        del bytes_pair_to_pretoken_positions[most_frequent_pair]

    def get_merge_as_list_of_tuples(self) -> list[tuple[bytes, bytes]]:
        """Get the list of merges as a list of tuples."""
        return [(pair.first, pair.second) for pair in self._merges]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        spelcial_tokens: list[str] | None = None,
    ) -> "BpeTokenizer":
        """Load a BPE tokenizer from saved vocab and merges files.

        Args:
            vocab_path (str): Path to the pickled vocab file.
            merges_path (str): Path to the pickled merges file.
            spelcial_tokens (list[str] | None): List of special tokens to include in the tokenizer.
        Returns:
            BpeTokenizer: The loaded BPE tokenizer.
        """
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab=vocab, merges=merges, special_tokens=spelcial_tokens)

    def _init_for_encoding(self) -> None:
        """Initialize any data structures needed for encoding."""
        self._merges_to_idx = {merge: idx for idx, merge in enumerate(self._merges)}
        self._inverted_vocab = {
            token: token_id for token_id, token in self._vocab.items()
        }
        self._longest_vocab_length = max(len(token) for token in self._vocab.values())

    def encode(self, text: str, show_progress: bool = False) -> list[int]:
        """Encode the input text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: List of token IDs representing the encoded text.
        """
        if (not self._merges_to_idx) or (not self._inverted_vocab):
            self._init_for_encoding()
        output = []
        # Split by special tokens wrapped in capturing groups to keep the delimiters.
        split_regex = b"(" + self._special_tokens_split_regex + b")"
        all_split_parts = re.split(split_regex, text.encode("utf-8"))
        if show_progress:
            all_split_parts = tqdm(all_split_parts)
        for split_part in all_split_parts:
            if not split_part:
                continue
            if split_part in self._special_tokens:
                token_id = self._inverted_vocab[split_part]
                output.append(token_id)
                continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                token_ids = self._encode_one_pretoken(match.group(0))
                output.extend(token_ids)
        return output

    @profile
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of texts into an iterator of token IDs."""
        if (not self._merges_to_idx) or (not self._inverted_vocab):
            self._init_for_encoding()
        leftover = b""
        input_it = iter(iterable)
        while True:
            try:
                new_text = next(input_it).encode("utf-8")
            except StopIteration:
                break
            current_chunk = leftover + new_text
            lookahead = b""
            while len(lookahead) < self._longest_vocab_length:
                try:
                    lookahead += next(input_it).encode("utf-8")
                except StopIteration:
                    break
            for result in self._encode_one_chunk(current_chunk, lookahead):
                if isinstance(result, int):
                    yield result
                else:
                    assert isinstance(result, bytes)
                    leftover = result
                    break
        if leftover:
            for result in self._encode_one_chunk(leftover, b""):
                if isinstance(result, int):
                    yield result
                else:
                    assert result == b""

    def _encode_one_chunk(
        self,
        chunk: bytes,
        lookahead: bytes,
    ) -> Iterator[int | bytes]:
        """Encode a single chunk of bytes into a list of token IDs.

        Args:
            chunk (bytes): The input bytes chunk to encode.
            lookahead (bytes): Additional bytes to look ahead for tokenization. This is used to
                handle  cases where a token may span across chunk boundaries.

        Returns:
            Iterator[int | bytes]: First yield tokens until `chunk` is fully tokenized. Some part of
                `lookahead` will also be tokenized. Then yield the remaining part of `loohahead`.
        """
        num_bytes_tokenized = 0
        main_chunk_size = len(chunk)
        combined_chunk = chunk + lookahead
        for split_part in re.split(
            b"(" + self._special_tokens_split_regex + b")", combined_chunk
        ):
            if not split_part:
                continue
            if split_part in self._special_tokens:
                token_id = self._inverted_vocab[split_part]
                yield token_id
                num_bytes_tokenized += len(split_part)
                if num_bytes_tokenized >= main_chunk_size:
                    yield combined_chunk[num_bytes_tokenized:]
                    return
                continue
            for match in self._pretokenization_regex_pattern.finditer(split_part):
                pretoken = match.group(0)
                token_ids = self._encode_one_pretoken(pretoken)
                yield from token_ids
                num_bytes_tokenized += len(pretoken)
                if num_bytes_tokenized >= main_chunk_size:
                    yield combined_chunk[num_bytes_tokenized:]
                    return
        assert False, f"chunk: {chunk}, lookahead: {lookahead}"

    def _encode_one_pretoken(self, pretoken: bytes) -> list[int]:
        """Encode a single pretoken into a list of token IDs.

        Args:
            pretoken (bytes): The pretoken to encode.

        Returns:
            list[int]: List of token IDs representing the encoded pretoken.
        """
        # Initialization.
        # Head's bytes pair always has b"" as first byte, so the head's bytes pair will never be
        # merged. This simplifies the logic when collecting final tokens.
        (
            head_byte_pair_node,
            min_merge_idx_heap,
        ) = self._init_pretoken_for_encoding(pretoken)
        # BPE merging.
        while min_merge_idx_heap:
            _, bytes_pair, bytes_pair_node = heapq.heappop(min_merge_idx_heap)
            if bytes_pair != bytes_pair_node.bytes_pair:
                # Check if this bytes pair is still valid. `bytes_pair` could be stale because the
                # actual bytes pair as in `bytes_pair_node` could have been updated when its
                # neighboring nodes are merged.
                continue
            self._merge_one_node_for_encoding(bytes_pair_node, min_merge_idx_heap)
        # Collect the final tokens.
        return self._bytes_pair_linked_list_to_token_ids(head_byte_pair_node)

    def _merge_one_node_for_encoding(
        self,
        node: BytesPairListNode,
        min_merge_idx_heap: list[tuple[int, BytesPair, BytesPairListNode]],
    ) -> None:
        if node.prev is None:
            raise ValueError(
                "Trying to merge a node that has `prev == None`, which can only be true for"
                " the hdead node, which cannot be merged."
            )

        new_token = node.bytes_pair.first + node.bytes_pair.second
        # Update the previous node.
        node.prev.next = node.next
        new_bytes_pair = BytesPair(
            first=node.prev.bytes_pair.first,
            second=new_token,
        )
        node.prev.bytes_pair = new_bytes_pair
        if new_bytes_pair in self._merges_to_idx:
            merge_idx = self._merges_to_idx[new_bytes_pair]
            heapq.heappush(min_merge_idx_heap, (merge_idx, new_bytes_pair, node.prev))
        # Update the next node's previous pointer and bytes pair.
        if node.next is None:
            return
        node.next.prev = node.prev
        new_bytes_pair = BytesPair(
            first=new_token,
            second=node.next.bytes_pair.second,
        )
        node.next.bytes_pair = new_bytes_pair
        if new_bytes_pair in self._merges_to_idx:
            merge_idx = self._merges_to_idx[new_bytes_pair]
            heapq.heappush(min_merge_idx_heap, (merge_idx, new_bytes_pair, node.next))

    def _init_pretoken_for_encoding(
        self,
        pretoken: bytes,
    ) -> tuple[
        BytesPairListNode,
        list[tuple[int, BytesPair, BytesPairListNode]],
    ]:
        """Initialize any data structures needed for encoding."""
        if not pretoken:
            raise ValueError("Input pretoken must be non-empty bytes.")
        byte_sequence = list(pretoken)
        min_merge_idx_heap: list[tuple[int, BytesPair, BytesPairListNode]] = []
        head_bytes_pair_node = BytesPairListNode(
            bytes_pair=BytesPair(first=b"", second=bytes([byte_sequence[0]])),
            prev=None,
            next=None,
        )
        prev_node = head_bytes_pair_node
        for i in range(1, len(byte_sequence)):
            bytes_pair = BytesPair(
                first=bytes([byte_sequence[i - 1]]),
                second=bytes([byte_sequence[i]]),
            )
            current_node = BytesPairListNode(
                bytes_pair=copy.deepcopy(bytes_pair), prev=prev_node
            )
            prev_node.next = current_node
            if bytes_pair in self._merges_to_idx:
                merge_idx = self._merges_to_idx[bytes_pair]
                min_merge_idx_heap.append((merge_idx, bytes_pair, current_node))
            prev_node = current_node
        heapq.heapify(min_merge_idx_heap)
        return head_bytes_pair_node, min_merge_idx_heap

    def _bytes_pair_linked_list_to_token_ids(
        self,
        head_byte_pair: BytesPairListNode | None,
    ) -> list[int]:
        """Convert a linked list of byte pairs to a list of token ids."""
        tokens: list[int] = []
        # Traverse the linked list and collect tokens.
        current_node = head_byte_pair
        while current_node is not None:
            # Head's bytes pair always has b"" as first byte, so we skip it.
            tokens.append(self._inverted_vocab[current_node.bytes_pair.second])
            current_node = current_node.next
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids (list[int]): List of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        byte_chunks = [self._vocab[token_id] for token_id in token_ids]
        decoded_string = b"".join(byte_chunks).decode("utf-8", errors="replace")
        return decoded_string
