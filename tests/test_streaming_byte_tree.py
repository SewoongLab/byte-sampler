from collections import ChainMap
from pathlib import Path
import sys

import pytest
import torch

sys.setrecursionlimit(max(5000, sys.getrecursionlimit()))
from transformers import PreTrainedTokenizerFast

from byte_sampling.byte_conditioning import ByteConditioning


TOKENIZER_ROOT = Path(__file__).resolve().parents[1] / "tokenizers"
TOKENIZER_PATHS = sorted(path for path in TOKENIZER_ROOT.iterdir() if path.is_dir())

if not TOKENIZER_PATHS:
    pytest.skip("No tokenizers found under tokenizers/", allow_module_level=True)


TEST_STRINGS = {
    "hello": "Hello world!",
    "punct": "— 'mix' of …?! ()[]{} ...",
    "contractions1": "I've i'd we'v 'hi' You''ve",
    "contractions2": "whomst'd've'ly'yaint'nt'ed'ies's'y'es",
    "quotes": "can't vs can’t; 'smart' “quotes”",
    "whitespace": "a b\tc\nd\r\ne\n\ng\u00a0h\u2009i\u200b",
    "newline1": "a\nb\r\nc\rd\u0085e\u2028f\u2029g\n",
    "newline2": "X\n\n\nY\r\n\r\nZ\r\r\u2028\u2029",
    "lead_spaces": "a a  a    a     a      a        a",
    "edge_space1": " a ",
    "edge_space2": "  a  ",
    "edge_space3": "   a   ",
    "digits1": "123 1234 12345 123456 1234567",
    "digits2": "0١۲३४५६७८९ २३,४५६.७८\u202f๙₀%",
    "digits3": "١٢٣ ١٢٣٤ ١٢٣٤٥ ١٢٣٤٥٦ ١٢٣٤٥٦٧",
    "digits4": "１２３ １２３４ １２３４５ １２３４５６",
    "cjmix": "你好，world！中文Mixed漢字カタカナかな混合123",
    "ko": "가각각ㄱㅏㄱ·테스트값·밖·꽃 vs 값/박/꽃",
    "ja": "東京タワー々・ｶﾞ/が/が 半角ｶﾀｶﾅｰ/全角カタカナー/ゝ々",
    "zh": "简繁混用：后/後，发/發；空格\u3000測試。《中文·Test》，——全角！",
    "vi": "Đắk Lắk: phở, bún, gỏi, Sài Gòn! Quốc ngữ / Quốc ngữ ó̂/ự",
    "bn": "র্কি ক্ষ ক\u09cd\u200cষ ক\u09cd\u200dষ ১২৩ ৳। \u09dc\u0981\u0983\u09cd",
    "th": "กำลังทดสอบ",
    "url": "https://a.b/c?x=1&y=2#z",
    "path": "C:\\a b\\c\\d.txt ~/src/.env.d/.config.yml",
    "emoji": "👨‍👩‍👧‍👦🏳️‍🌈🇺🇸a✊🏿 ☝️+🙂=☹️?",
    # "special": "<|endoftext|><s> </s>",
    "html": "<script>alert('&<>\"');</script><!--a--><img src=x alt='&>'/>",
    "math": "x²+y²=1; α→β; ±∞; 1e−9≈0 f′(x)=∫₀^∞ e^{−x²}dx ∈ ℝ",
    "camel": "parseHTTPServerURL2JSONv2BetaXiOS15Pro",
    "rtl": "\u202bעברית\u202c abc\u200fעברית\u200e123",
    "control": "control\x00a\x1fb\x7fccontrol",
    "combine": "e\u0301é n\u0303\u0301 vs ñ",
    "modifier": "o\u02bcclock ya\u02e3",
    "indic": "क\u094d\u200cष क\u094dष",
    # New tests
    "bom": "\ufeffBOM-start",
    # "invalid": "\ud800\udfff\u0301\u200d\u202e\ufeff",
    "flags": "🇺🇸🇦🇬🇧🏳️‍🌈🏴‍☠️🇨🇳🇪🇺🇦",
    "c": "if (x->y) {\n    s += '\\n'; /* c */}\n//eol",
    "py": "def f_(x: int=0, *a, **k) -> str:\n    return f'{x:08x}'",
    "case": "ǅen’s ǅEN",
}


@pytest.fixture(scope="session", params=TOKENIZER_PATHS, ids=lambda path: path.name)
def byte_conditioning(request) -> ByteConditioning:
    path: Path = request.param
    if not hasattr(byte_conditioning, "_cache"):
        byte_conditioning._cache = {}

    cached = byte_conditioning._cache.get(path)
    if cached is None:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        cached = ByteConditioning(model_or_dir=None, tokenizer=tokenizer)
        token_slicer_special = ByteConditioning.TokenSlicer(
            ChainMap(
                cached.vrev,
                {
                    tid: at.content.encode("utf-8")
                    for tid, at in cached.tokenizer.added_tokens_decoder.items()
                },
            ),
            "cpu",
        )
        cached.token_slicer_special = token_slicer_special
        byte_conditioning._cache[path] = cached

    return cached


def _token_bytes(bc: ByteConditioning, tid: int) -> bytes:
    if (result := bc.vrev.get(tid)) is not None:
        return result

    special = bc.tokenizer.added_tokens_decoder
    if (result := special.get(tid)) is not None:
        return result.content.encode("utf-8")

    assert False, "could not find tid {tid} in vocabulary!"


@pytest.mark.parametrize(
    "case_name,text", TEST_STRINGS.items(), ids=list(TEST_STRINGS.keys())
)
def test_streaming_byte_tree_invariants(
    byte_conditioning: ByteConditioning,
    case_name: str,
    text: str,
    inclusive=True,
    filter_tensors=False,
):
    bc = byte_conditioning

    # we assume any prefilled text is normalized and
    # that the model will not sample unnormalized text
    if bc.btok.normalizer:
        text = bc.btok.normalizer.normalize_str(text)

    sbt = bc.get_streaming_byte_tree()
    text_bytes = text.encode("utf-8")
    trunk = []
    trunk_bytes = 0
    canonical = bc.tokenizer.encode(text, add_special_tokens=False)
    if bc.tokenizer.decode(canonical) != text:
        pytest.skip("tokenizer must itself be consistent")
    # assert bc.tokenizer.decode(canonical) == text, "tokenizer must itself be consistent"

    for byte_idx, byte in enumerate(text_bytes):
        trunk_new = sbt.push(byte)
        tree = sbt.eval_tree(inclusive=inclusive, filter_tensors=filter_tensors)

        # Test 1 (consistency): check that trunk_new matches canonical
        assert len(trunk) + len(trunk_new) <= len(
            canonical
        ), f"{byte_idx}: trunk cannot extend past prefix"
        assert (
            canonical[len(trunk) : len(trunk) + len(trunk_new)] == trunk_new
        ), f"{byte_idx}: trunk must match prefix"
        trunk.extend(trunk_new)
        trunk_bytes += sum(len(_token_bytes(bc, tid)) for tid in trunk_new)

        # Test 2 (completeness): check that the canonical tokenization is in the tree
        pointer, node_idx, node_bytes = tree, len(trunk), trunk_bytes
        while node_bytes < byte_idx + inclusive:
            assert node_idx < len(canonical)
            next_tid = canonical[node_idx]
            next_len = len(_token_bytes(bc, next_tid))

            # TODO: make this work for inclusive=False
            if node_bytes + next_len > byte_idx + inclusive:
                # byte_idx + 1 is the length of the current prefix
                assert next_tid in pointer.get(
                    None, ()
                ), f"{byte_idx}: must have canonical path as a leaf"

                # We've read up to the end, so we are done now.

                break

            assert (
                next_tid in pointer
            ), f"{byte_idx}: tree internal nodes must contain canonical path"
            pointer = pointer[next_tid]
            node_idx += 1
            node_bytes += next_len

        # Test 3 (correctness): check that all leaves of the tree match the prefix
        def traverse_tree(node, node_bytes):
            for tid, subtree in node.items():
                if tid is None:
                    # If we are at the end of the prompt with
                    # inclusive=True, then any continuation is valid.
                    if node_bytes == byte_idx + 1:
                        assert inclusive
                        continue

                    # Now check that leaves match the prefix
                    prefix = text_bytes[node_bytes : byte_idx + 1]
                    valid_tids = bc.token_slicer_special.query(prefix)
                    assert (
                        torch.isin(subtree, valid_tids).all().item()
                    ), f"{byte_idx}: tree leaves must match prefix {prefix!r}: {subtree}"

                    # Also check that the other tids aren't duplicated
                    other_tids = [t for t in node.keys() if t is not None]
                    assert not (
                        other_tids
                        and torch.isin(
                            torch.tensor(other_tids, device=subtree.device), subtree
                        )
                        .any()
                        .item()
                    ), "Duplicated tid in leaf and branch"

                    continue

                next_tok = _token_bytes(bc, tid)
                next_node_bytes = node_bytes + len(next_tok)
                assert (
                    next_node_bytes <= byte_idx + inclusive
                ), f"{byte_idx}: tree cannot extend past prefix"
                assert (
                    next_tok == text_bytes[node_bytes:next_node_bytes]
                ), f"{byte_idx}: tree internal nodes must match prefix"
                traverse_tree(subtree, next_node_bytes)

        traverse_tree(tree, trunk_bytes)
