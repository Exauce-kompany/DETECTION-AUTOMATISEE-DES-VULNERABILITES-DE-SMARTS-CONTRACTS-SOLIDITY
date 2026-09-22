"""Shared, versioned preprocessing for dataset building and production inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "solidity-v3.1"
LEXER = re.compile(
    r'(?P<comment>//[^\r\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<string>"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\')'
    r'|(?P<number>0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?)'
    r'|(?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*)'
    r'|(?P<operator>>>=|<<=|>>=|\*\*=|==|!=|<=|>=|&&|\|\||\+\+|--|=>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|\*\*)'
    r'|(?P<symbol>\S)'
)
KEYWORDS = set("""abstract address after alias anonymous apply as assembly auto bool break
byte bytes calldata case catch constant constructor continue contract copyof default define
delete do else emit enum error event external false final fixed for from function global
if immutable implements import in indexed inline interface internal is let library macro
mapping match memory modifier mutable new null of override payable pragma private promise
public pure reference relocatable return returns revert sealed sizeof static storage string
struct super supports switch this throw true try type typedef typeof ufixed unchecked using
var view virtual while uint int wei gwei ether seconds minutes hours days weeks years
msg sender value data sig tx origin gasprice block timestamp number difficulty prevrandao
coinbase gaslimit basefee chainid blockhash gasleft assert require keccak256 sha256 ripemd160
ecrecover addmod mulmod selfdestruct suicide abi encode encodePacked encodeWithSelector
encodeWithSignature decode balance code codehash call callcode delegatecall staticcall
send transfer push pop length selector add sub mul div mod sload sstore mload mstore
mstore8 calldataload calldatacopy returndatasize returndatacopy revert stop log0 log1
log2 log3 log4 create create2 origin caller callvalue eq lt gt slt sgt iszero and or
xor not shl shr sar byte exp signextend invalid gas address selfbalance pc msize
solidity experimental ABIEncoderV2 unicode hex now""".split())
TYPE_NAME = re.compile(r"(?:u?int\d*|bytes\d*|u?fixed\d*x?\d*)\Z")


def load_config(path=None):
    path = Path(path) if path else ROOT / "config/v3.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["preprocessing_version"] != VERSION:
        raise ValueError("Preprocessing version incompatible with this code.")
    if config["window_size"] < 1 or config["vocabulary_size"] < 259:
        raise ValueError("Invalid sequence/vocabulary configuration.")
    return config


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def code_from_sample(sample):
    code = sample.get("context", "")
    if isinstance(code, list) and all(isinstance(line, str) for line in code):
        return "\n".join(code)
    if isinstance(code, str):
        return code
    raise ValueError("context must contain source text or a list of source lines")


def tokenize(code, normalize_identifiers=True):
    """Discard comments without modifying quoted strings or silently truncating code.

    User identifiers are renamed consistently within each input. This removes
    annotation-like names (bug_reentrancy_1, etc.) without losing repeated-name
    relationships. Language keywords, primitive types and known builtins remain.
    """
    if not isinstance(code, str):
        raise ValueError("Le code doit être une chaîne de caractères.")
    tokens, identifiers = [], {}
    for match in LEXER.finditer(code):
        kind, value = match.lastgroup, match.group()
        if kind == "comment":
            continue
        if normalize_identifiers:
            if kind == "string":
                value = "<STRING>"
            elif kind == "identifier" and value not in KEYWORDS and not TYPE_NAME.fullmatch(value):
                value = identifiers.setdefault(value, f"<ID_{len(identifiers)}>")
            elif kind == "number":
                value = value.replace("_", "").lower()
                if value.startswith("0x") and len(value) >= 42:
                    value = "<ADDRESS>"
        tokens.append(value)
    return tokens


def structural_tokens(tokens):
    """Broader clone-family key; never use this to decide/rewrite labels."""
    return ["<NUMBER>" if re.match(r"^\d", token) else token for token in tokens]


def build_vocabulary(token_sequences, limit):
    if limit < 259:
        raise ValueError("Vocabulary must reserve PAD, UNK, token boundary and 256 byte values.")
    counts = Counter(token for sequence in token_sequences for token in sequence)
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for value in range(256):
        vocab[f"<BYTE_{value:02x}>"] = len(vocab)
    vocab["<TOKEN_END>"] = len(vocab)
    for token in ordered:
        if len(vocab) >= limit:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocabulary):
    if not tokens:
        raise ValueError("Aucun token Solidity exploitable après retrait des commentaires.")
    ids = []
    for token in tokens:
        if token in vocabulary:
            ids.append(vocabulary[token])
        else:
            # An explicit start/end boundary makes this representation injective:
            # unseen identifiers/literals cannot collapse to one common UNK input.
            ids.append(1)
            ids.extend(vocabulary[f"<BYTE_{value:02x}>"] for value in token.encode("utf-8"))
            ids.append(vocabulary["<TOKEN_END>"])
    return np.asarray(ids, dtype=np.int32)


def as_windows(ids, window_size):
    """Cover every token. Padding is the only information added."""
    ids = np.asarray(ids, dtype=np.int32)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("Expected a nonempty one-dimensional token sequence.")
    windows = np.zeros(((len(ids) + window_size - 1) // window_size, window_size), dtype=np.int32)
    windows.flat[:len(ids)] = ids
    return windows


def prepare_code(code, vocabulary, window_size):
    tokens = tokenize(code)
    ids = encode(tokens, vocabulary)
    windows = as_windows(ids, window_size)
    unknown = int((ids == 1).sum())
    return windows, {
        "tokens_detected": len(tokens), "tokens_used": len(tokens),
        "truncated": False, "windows": len(windows), "window_size": window_size,
        "encoded_tokens": len(ids), "unknown_encoding": "lossless_utf8_byte_fallback",
        "unknown_tokens": unknown, "unknown_rate": unknown / len(tokens),
        "preprocessing_version": VERSION, "comments_removed": True,
    }
