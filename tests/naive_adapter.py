def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    """Train a BPE tokenizer with vocab: [special tokens] + [256 bytes] + [merge tokens]."""

    with open(input_path, "rb") as f:
        text = f.read()

    special_token_map = {}
    for idx, tok in enumerate(special_tokens):
        placeholder = f"<@_SPECIAL{idx}_@>"
        text = text.replace(tok, placeholder)
        special_token_map[placeholder] = tok


    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    units = []
    for part in re.findall(PAT, text):
        if part in special_token_map:
            units.append(special_token_map[part])
        elif part:
            units.append(part)

    token_seqs = []
    for unit in units:
        if unit in special_tokens:
            token_seqs.append([unit.encode('utf-8')])
        else:
            token_seqs.append([bytes([b]) for b in unit.encode('utf-8')])

    # === 构建vocab（special->bytes->merge tokens） ===
    vocab = {}

    # 1. special tokens
    for i, tok in enumerate(special_tokens):
        vocab[i] = tok.encode('utf-8')

    # 2. 256 byte chars，必须补齐全部
    byte_chars = [bytes([i]) for i in range(256)]
    for i, b in enumerate(byte_chars):
        vocab[len(special_tokens) + i] = b

    # 3. 后续的 merge token id 按 vocab 长度递增分配
    merges: List[Tuple[bytes, bytes]] = []

    # ==== BPE merge循环 ====
    # 新增的合成token id 从 vocab_offset 往后排
    vocab_offset = len(vocab)
    while len(vocab) < vocab_size:
        pairs = Counter()
        for seq in token_seqs:
            # 跳过special token单独unit
            if len(seq) == 1 and seq[0] in [st.encode('utf-8') for st in special_tokens]:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pairs[pair] += 1
        if not pairs:
            break
        
        max_freq = max(pairs.values())
        max_pairs = [pair for pair, cnt in pairs.items() if cnt == max_freq]
        most_common = max(max_pairs)

        merges.append(most_common)
        merged_token = most_common[0] + most_common[1]

        new_token_seqs = []
        for seq in token_seqs:
            if len(seq) == 1 and seq[0] in [st.encode('utf-8') for st in special_tokens]:
                new_token_seqs.append(seq)
            else:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == most_common:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_token_seqs.append(new_seq)
        token_seqs = new_token_seqs

        if merged_token not in vocab.values():
            vocab[len(vocab)] = merged_token

    return vocab, merges