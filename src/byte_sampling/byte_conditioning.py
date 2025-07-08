import bisect
import heapq
import itertools as it
import time
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Optional, Self, Union

import numpy as np
import regex
import torch
import simdjson as json
from ahocorasick_rs import AhoCorasick, MatchKind
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import DoublyLinkedList, bytes_to_unicode, scatter_logsumexp, build_trie
from .radix_cache import RadixCacheManager

class BaseBytewiseBatchSampler(ABC):
    @abstractmethod
    def add_context(self, prompts: list[Union[str, bytes]]):
        pass

    @abstractmethod
    def add_special_context(self, prompts: list[list[int]]):
        pass

    @abstractmethod
    def get_dists(self) -> torch.Tensor:
        pass


class ByteConditioning(object):
    class TokenIndexerCache:
        """Class to cache the arrays of nth bytes of each token"""

        def __init__(self, tcs):
            self.tcs = tcs
            self.device = tcs.model.device
            self.cache = {}

        def get_cache(self, idx):
            assert idx >= 0
            result = [256] * self.tcs.tokenizer.vocab_size
            for i in range(self.tcs.tokenizer.vocab_size):
                if i not in self.tcs.vrev:
                    continue
                tok = self.tcs.vrev[i]
                if idx < len(tok):
                    result[i] = tok[idx]
            return torch.tensor(result, device=self.device)

        def get(self, i: int):
            if i not in self.cache:
                self.cache[i] = self.get_cache(i)
            return self.cache[i]

    def __init__(
        self,
        model_or_dir,
        tokenizer=None,
        load_kwargs=None,
        skip_model=False,
        skip_preprocess_merges=False,
    ):
        assert not (isinstance(model_or_dir, str) and tokenizer is not None)
        if tokenizer is not None:
            self.model = model_or_dir
            self.tokenizer = tokenizer
        else:
            load_kwargs = (
                dict(device_map="auto", torch_dtype=torch.bfloat16) if load_kwargs is None else load_kwargs
            )
            if not skip_model:
                self.model = AutoModelForCausalLM.from_pretrained(model_or_dir, **load_kwargs)
            else:
                self.model = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_dir)

        self.device = self.model.device if self.model is not None else torch.get_default_device()
        self.dtype = self.model.dtype if self.model is not None else torch.get_default_dtype()

        self.btok = self.tokenizer.backend_tokenizer
        self.bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self.eos = self.tokenizer.eos_token_id or self.tokenizer.bos_token_id
        self.pad = self.tokenizer.pad_token_id or self.bos

        # this is awful...
        raw_state = json.loads(self.tokenizer.backend_tokenizer.__getstate__())

        # handle special tokens
        added_tokens = raw_state["added_tokens"]
        added_token_ids = set(tok["id"] for tok in added_tokens)

        added_tokens = raw_state["added_tokens"]
        for tok in added_tokens:
            assert not tok.get("single_word"), f"{tok}"
            assert not tok.get("lstrip"), f"{tok}"
            assert not tok.get("rstrip"), f"{tok}"

        self.split_trie = AhoCorasick(
            [tok["content"] for tok in added_tokens if not tok["normalized"]],
            matchkind=MatchKind.LeftmostLongest,
        )
        self.split_ids = [tok["id"] for tok in added_tokens if not tok["normalized"]]

        self.split_normalized_trie = AhoCorasick(
            [tok["content"] for tok in added_tokens if tok["normalized"]],
            matchkind=MatchKind.LeftmostLongest,
        )
        self.split_normalized_ids = [tok["id"] for tok in added_tokens if tok["normalized"]]
        if self.btok.normalizer is not None:
            assert not self.split_normalized_ids

        # extract the vocabulary
        self.btu = bytes_to_unicode()
        self.utb = {c: b for b, c in self.btu.items()}
        self.vocab = {
            bytes(self.utb[c] for c in tok): tid
            for tok, tid in raw_state["model"]["vocab"].items()
            if tid not in added_token_ids
        }
        self.vrev = {tid: tok for tok, tid in self.vocab.items()}

        # check that we support the pretokenizer
        digit3_right_just_re = r"(?=(\d{3})+(?!\d))"
        whitespace_lookahead_re = r"\s+(?!\S)"
        whitespace_newline_exception_re = r"\s*[\r\n]+"
        contraction_merge_re = "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        contraction_bridge_re = r"[^\r\n\p{L}\p{N}]?\p{L}+"

        self.has_digit3_right_just = False
        self.has_whitespace_lookahead = False
        self.has_contraction_discontinuity = False
        self.has_ignore_merges = raw_state["model"].get("ignore_merges")

        has_bytelevel = False
        for pt in raw_state["pre_tokenizer"]["pretokenizers"]:
            if pt["type"] == "Digits" and not pt["individual_digits"]:
                pass
            elif pt["type"] == "ByteLevel" and not pt["add_prefix_space"] and not pt["use_regex"]:
                has_bytelevel = True
            elif pt["type"] == "Split" and (
                (pt["behavior"] == "Isolated" and not pt["invert"])
                or (pt["behavior"] == "Removed" and pt["invert"])
            ):
                # TODO: This is not "correct"
                regexes = pt["pattern"]["Regex"].split("|")
                if digit3_right_just_re in regexes:
                    self.has_digit3_right_just = True
                    for i in range(0, 1000):
                        assert str(i).encode() in self.vocab, f"Digit {i} is missing from the vocab!"

                if whitespace_lookahead_re in regexes:
                    self.has_whitespace_lookahead = True
                    assert whitespace_newline_exception_re in regexes
                    assert regexes.index(whitespace_newline_exception_re) < regexes.index(whitespace_lookahead_re)

                if contraction_merge_re in regexes and contraction_bridge_re not in regexes:
                    self.has_contraction_discontinuity = True
            else:
                print(f"Unknown pretokenizer {pt}")
                # raise NotImplementedError(f"Unknown pretokenizer {pt}")
                pass

        assert not self.has_contraction_discontinuity, "Don't support this right now!"
        assert has_bytelevel, "Tokenizer must be byte-level!"

        # extract the merges
        self.merges = []
        self.merge_map = {}
        self.order_map = {}
        for i, merge in enumerate(raw_state["model"]["merges"]):
            left, right = merge
            l = self.vocab[bytes(self.utb[c] for c in left)]
            r = self.vocab[bytes(self.utb[c] for c in right)]
            t = self.vocab[bytes(self.utb[c] for c in left + right)]
            self.merges.append((l, r, t))

        # remove merges that will never be used
        if not skip_preprocess_merges:
            self.merges = self._preprocess_merges(self.merges)

        for i, (l, r, t) in enumerate(self.merges):
            self.merge_map[l, r] = t
            self.merge_map[t] = (l, r)
            self.order_map[l, r] = i
            self.order_map[t] = i

        self.left_map = {}
        for i, (l, r, lr) in enumerate(self.merges):
            self.left_map.setdefault(l, []).append(i)
        self.walk_cache = {}
        self.walk_cache_min_size = 1000
        self.vtrie = build_trie(self.vocab)
        for tok, tid in self.vocab.items():
            pointer = self.vtrie
            for b in tok:
                pointer = pointer[b]
            pointer[None] = tid

        if self.model:
            self.token_index_cache = self.TokenIndexerCache(self)

    def _preprocess_merges(self, merges):
        # merge_multi_map = {}
        order_map = {}
        merge_map = {}
        for i, (l, r, lr) in enumerate(merges):
            # merge_multi_map.setdefault(lr, []).append((l, r))
            order_map[l, r] = i
            merge_map[l, r] = lr

        resolve_map = {}

        def resolve_subtokens(buffer: bytes, verbose=False):
            ll = DoublyLinkedList([(i, self.vocab[bytes([b])]) for i, b in enumerate(buffer)])
            q = []

            def maybe_add_merge(node):
                if nnode := node.n:
                    (i, a), (_, b) = node.obj, nnode.obj
                    if (order := order_map.get((a, b))) is not None:
                        heapq.heappush(q, (order, i, node, a, b))

            # add the base merges
            for node in ll:
                maybe_add_merge(node)

            while q:
                order, i, node, a, b = heapq.heappop(q)
                if (nnode := node.n) is None:
                    continue

                (i, found_a), (_, found_b) = node.obj, nnode.obj
                if not (a == found_a and b == found_b):
                    continue

                # if verbose:
                #     print('merge', order, i)
                ab = merge_map[(a, b)]
                if ab in resolve_map:
                    # print(ab, a, b, resolve_map[ab])
                    assert resolve_map[ab] == (a, b)
                else:
                    resolve_map[ab] = a, b
                newnode = ll.Node((i, ab), node.p, nnode.n)
                node.p = node.n = nnode.p = nnode.n = None

                if newnode.p:
                    newnode.p.n = newnode
                    maybe_add_merge(newnode.p)
                else:
                    ll.head = newnode

                if newnode.n:
                    newnode.n.p = newnode
                    maybe_add_merge(newnode)
                else:
                    ll.tail = newnode

            return [node.obj[1] for node in ll]

        self.unreachable_tokens = set()
        skip = 0
        for tbytes, tid in sorted(self.vocab.items(), key=lambda pair: len(pair[0]), reverse=True):
            if len(tbytes) == 1:
                continue
            if tid not in resolve_map:
                out = resolve_subtokens(tbytes)
                if tid not in resolve_map:
                    self.unreachable_tokens.add(tid)
                    assert out != [tid], f"{tid}"
            else:
                skip += 1

        order_cache = {}

        def get_order(tid):
            if tid in self.unreachable_tokens:
                return (float("inf"),)
            if (lr := resolve_map.get(tid)) is None:
                return (-1,)
            if (cached := order_cache.get(tid)) is not None:
                return cached
            l, r = lr
            ordlr = order_map[l, r]
            rmax = *rpath, rlast = max(get_order(l), get_order(r))
            new = (*rmax, ordlr) if ordlr <= rlast else (*rpath, ordlr)
            while len(new) >= 2 and new[-1] > new[-2]:
                new = (*new[:-2], new[-1])
            order_cache[tid] = new
            return new

        new_merges = []
        for tid in sorted(self.vrev, key=get_order):
            if tid not in self.unreachable_tokens and tid in resolve_map:
                new_merges.append((*resolve_map[tid], tid))

        return new_merges

    @classmethod
    def _walk_vtrie(cls, vtrie):
        if None in vtrie:
            yield vtrie[None]

        for b, subtrie in vtrie.items():
            if b is not None:
                yield from cls._walk_vtrie(subtrie)

    def _get_walk_cached(self, prefix: bytes):
        if prefix in self.walk_cache:
            return self.walk_cache[prefix]
        pointer = self.vtrie
        for b in prefix:
            if b not in pointer:
                return np.array([], int)
            pointer = pointer[b]
        candidates = np.fromiter(self._walk_vtrie(pointer), int)
        if len(candidates) > self.walk_cache_min_size:
            self.walk_cache[prefix] = candidates
        return candidates

    def _right_history(self, tok: int):
        hist = [(self.order_map.get(tok, -1), tok)]

        while True:
            pmerge = self.merge_map.get(hist[-1][1])
            if not pmerge:
                break
            parent = pmerge[1]
            hist.append((self.order_map.get(parent, -1), parent))

        return hist

    def _left_history(self, tok: int):
        hist = [(self.order_map.get(tok, -1), tok)]
        while True:
            pmerge = self.merge_map.get(hist[-1][1])
            if not pmerge:
                break
            parent = pmerge[0]
            hist.append((self.order_map.get(parent, -1), parent))

        return hist

    def _valid_adj(self, l: int, r: int):
        if (l, r) in self.merge_map:
            return False

        if l in self.unreachable_tokens or r in self.unreachable_tokens:
            return False

        # collect history of the right edge of l
        lrs = [(self.order_map.get(l, -1), 0, l, None)]
        while True:
            if not (pmerge := self.merge_map.get(lrs[-1][2])):
                break
            _, parent = pmerge
            lrs.append((self.order_map.get(parent, -1), 0, parent, None))

        # collect history of the left edge of r
        rls = [(self.order_map.get(r, -1), 1, None, r)]
        while True:
            if not (pmerge := self.merge_map.get(rls[-1][3])):
                break
            parent, _ = pmerge
            rls.append((self.order_map.get(parent, -1), 1, None, parent))

        # look for conflicts
        l, r = lrs[-1][2], rls[-1][3]
        lrs.pop()
        rls.pop()
        for j, inclusive, newl, newr in sorted(lrs + rls):
            if self.order_map.get((l, r), j + inclusive) < j + inclusive:
                return False
            l, r = newl or l, newr or r

        return True

    def _invalid_r_filtered(self, l: int, prefix: bytes):
        # collect history of the right edge of l
        lrs = self._right_history(l)

        visited = set()

        def propagate(idxs):
            for i in idxs:
                tid = self.merges[i][-1]
                text = self.vrev[tid]

                if text.startswith(prefix):
                    visited.add(tid)
                elif prefix.startswith(text):
                    pass
                else:
                    continue

                visited.add(tid)
                if right_merges := self.left_map.get(tid):
                    propagate(right_merges)

        def propagate_base(idxs):
            for i in idxs:
                tid = self.merges[i][1]
                text = self.vrev[tid]

                if text.startswith(prefix):
                    visited.add(tid)
                elif prefix.startswith(text):
                    pass
                else:
                    continue

                if right_merges := self.left_map.get(tid):
                    start = bisect.bisect_left(right_merges, i)
                    propagate(right_merges[start:])

        oldi, oldl = lrs[-1]
        lrs.pop()
        for i, l in reversed(lrs):
            right_merges = self.left_map.get(oldl, [])
            end = bisect.bisect_left(right_merges, i)
            propagate_base(right_merges[:end])
            oldi, oldl = i, l

        propagate_base(self.left_map.get(l, []))

        return visited

    def _valid_r_filtered(self, l: int, prefix: bytes):
        invalid = np.fromiter(self._invalid_r_filtered(l, prefix), int)
        candidates = self._get_walk_cached(prefix)
        return np.setdiff1d(candidates, invalid, assume_unique=True)


    class StreamingBPE:
        @dataclass
        class Node:
            last_tid: int
            parent: Optional[Self]
            children: dict[int, Self]  # tid -> child
            trie: Optional[dict]
            trie_path: list

            def __repr__(self):
                return f"N({self.last_tid})"

            __str__ = __repr__

        def __init__(self, tcs: "ByteConditioning"):
            self.tcs = tcs
            self.reset()
            self.total_time = 0

        def reset(self):
            self.tree = self.Node(None, None, {}, self.tcs.vtrie, [])
            self.heads = [self.tree]
            self.last_head = self.tree

        def gc_node(self, node):
            if node.parent is not None and not node.children and node.trie is None:
                # print(f"removing node {node}")
                node.parent.children.pop(node.last_tid)
                self.gc_node(node.parent)

        def push(self, byte: int):
            assert isinstance(byte, int)
            new_heads = []
            fixed_tokens = []

            heads_created = []
            for head in self.heads:
                if byte not in head.trie:
                    # this head has "died"
                    head.trie = None
                    self.gc_node(head)
                    continue

                trie = head.trie = head.trie[byte]
                head.trie_path.append(byte)
                new_heads.append(head)
                if (newtid := trie.get(None)) is not None:
                    # if head.parent is None or self.tcs._valid_adj(head.last_tid, newtid):
                    if head.parent is None or self.tcs._valid_adj(head.last_tid, newtid):
                        newhead = self.Node(newtid, head, {}, self.tcs.vtrie, [])
                        self.last_head = newhead
                        head.children[newhead.last_tid] = newhead
                        new_heads.append(newhead)
                        heads_created.append(newhead)

            def trace_path(head):
                pathrev = []
                length = 0
                while True:
                    pathrev.append(head.last_tid)
                    if head.parent is None:
                        break
                    head = head.parent
                return (
                    sum(len(self.tcs.vrev.get(tid, b"")) for tid in pathrev),
                    pathrev[::-1],
                )

            assert (
                len(heads_created) <= 1 or self.tcs.has_ignore_merges
            ), f"got multiple paths to the same byte: {[trace_path(h) for h in heads_created]}"
            assert len(heads_created) >= 1, f"sequence ending in {bytes([byte])!r} cannot be tokenized"
            self.heads = new_heads

            while len(self.tree.children) == 1:
                if self.tree.trie is not None:
                    break

                new_root = next(iter(self.tree.children.values()))
                new_root.parent = None
                self.tree = new_root
                fixed_tokens.append(self.tree.last_tid)

            return fixed_tokens

        def split(self):
            pointer = self.last_head
            path_rev = []
            while pointer.parent is not None:
                path_rev.append(pointer.last_tid)
                pointer = pointer.parent
            self.reset()
            return path_rev[::-1]

        def fork(self):
            new = self.__class__(self.tcs)
            # deepcopy the tree but NOT the trie!
            new_heads = []
            new_last_head = None

            def copy_tree(node: self.Node, parent: Optional[self.Node] = None):
                nonlocal new_last_head
                newnode = self.Node(node.last_tid, parent, None, node.trie, copy(node.trie_path))
                if node in self.heads:
                    new_heads.append(newnode)
                if node is self.last_head:
                    new_last_head = newnode
                newnode.children = {tid: copy_tree(child, newnode) for tid, child in node.children.items()}
                return newnode

            new.tree = copy_tree(self.tree)
            new.heads = new_heads
            new.last_head = new_last_head

            return new

        def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
            if suffix:
                # for convenience
                self_copy = self.fork()
                copy_tokens = []
                for b in suffix:
                    copy_tokens.extend(self_copy.push(b))
                tree = self_copy.eval_tree(inclusive=inclusive)
                for tid in reversed(copy_tokens):
                    tree = {tid: tree}
                return tree

            def convert_tree(node: self.Node):
                converted_node = {}
                if node.trie is not None:
                    if filter_tensors:
                        # print(f"tree: {node.last_tid}, {bytes(node.trie_path) + suffix!r}")
                        valid_tokens = torch.from_numpy(
                            self.tcs._valid_r_filtered(node.last_tid, bytes(node.trie_path))
                            # self.tcs._get_walk_cached(bytes(node.trie_path)),
                        ).to(device=self.tcs.device)
                    else:
                        mask = torch.ones(
                            self.tcs.tokenizer.vocab_size,
                            dtype=torch.bool,
                            device=self.tcs.device,
                        )
                        for i, b in enumerate(node.trie_path):
                            mask &= self.tcs.token_index_cache.get(i) == b
                        valid_tokens = torch.arange(self.tcs.tokenizer.vocab_size, device=self.tcs.device)[mask]
                    if len(valid_tokens) > 0:
                        converted_node[None] = valid_tokens

                for tid, child in node.children.items():
                    subtree, was_last = convert_tree(child)
                    if subtree:
                        converted_node[tid] = subtree
                        if None in converted_node and was_last:
                            valid_tokens = converted_node[None]
                            converted_node[None] = valid_tokens[valid_tokens != tid]
                            if len(converted_node[None]) == 0:
                                converted_node.pop(None)

                if node is self.last_head:
                    if not inclusive:
                        return {}, True
                    else:
                        converted_node[None] = torch.arange(self.tcs.tokenizer.vocab_size, device=self.tcs.device)
                        # converted_node[None] = slice(len(self.tcs.vocab))

                return converted_node, node is self.last_head

            converted_tree, _ = convert_tree(self.tree)
            return converted_tree

    def get_streaming_bpe(self):
        return self.StreamingBPE(self)

    class StreamingCharPretok:
        def __init__(self, tcs: "ByteConditioning"):
            self.tcs = tcs
            self.sbpe = tcs.get_streaming_bpe()
            self.previous_sbpe = None
            self.held_tokens = None
            self.previous_held_tokens = None
            self.pretokenize = tcs.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
            self.buf = ""

        def push(self, char: str):
            if self.tcs.btok.normalizer is not None:
                char = self.tcs.btok.normalizer.normalize_str(char)
            self.buf += char
            pretokens = self.pretokenize(self.buf)
            fixed_tokens = []

            for (_, (_, aend)), (_, (bstart, _)) in it.pairwise(pretokens):
                assert aend == bstart, f"got gap in pretokens: {pretokens}"

            assert pretokens[0][1][1] >= len(self.buf) - 2, f"{pretokens}"
            if pretokens[0][1][1] == len(self.buf) - 2:
                # the special case
                assert self.previous_sbpe is not None, f"{pretokens}"
                self.held_tokens = None

                self.sbpe = self.previous_sbpe
                fixed_tokens.extend(self.previous_held_tokens)
                self.buf = self.buf[-2:]

                if len(pretokens) == 3:
                    fixed_tokens.extend(self.sbpe.split())
                    self.buf = self.buf[-1]

            else:
                # the normal case
                if self.held_tokens is not None:
                    fixed_tokens.extend(self.held_tokens)

                if pretokens[0][1][1] == len(self.buf) - 1:
                    fixed_tokens.extend(self.sbpe.split())
                    self.buf = self.buf[-1]

            # no matter what, these are no longer valid
            self.previous_sbpe = None
            self.held_tokens = None
            self.previous_held_tokens = None

            if self.tcs.has_whitespace_lookahead and regex.search(r"\s[^\S\r\n]$", self.buf):
                # TODO: double check this regex
                self.previous_sbpe = self.sbpe
                self.sbpe = self.sbpe.fork()
                self.held_tokens = []
                self.previous_held_tokens = []
                self.previous_held_tokens.extend(self.previous_sbpe.split())
                for b in char.encode():
                    self.held_tokens.extend(self.sbpe.push(b))
                    self.previous_held_tokens.extend(self.previous_sbpe.push(b))

                while (
                    self.held_tokens
                    and self.previous_held_tokens
                    and self.held_tokens[0] == self.previous_held_tokens[0]
                ):
                    # don't hit this often enough to switch to deque
                    fixed_tokens.append(self.held_tokens.pop(0))
                    self.previous_held_tokens.pop(0)
            else:
                for b in char.encode():
                    fixed_tokens.extend(self.sbpe.push(b))

            return fixed_tokens

        def split(self):
            self.buf = ""
            self.previous_sbpe = None
            self.held_tokens = None
            self.previous_held_tokens = None
            return self.sbpe.split()

        def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
            current_tree = self.sbpe.eval_tree(suffix=suffix, inclusive=inclusive, filter_tensors=filter_tensors)
            if self.previous_sbpe is None:
                assert self.held_tokens is None
                return current_tree

            for tid in reversed(self.held_tokens):
                current_tree = {tid: current_tree}

            previous_tree = self.previous_sbpe.eval_tree(
                suffix=suffix, inclusive=inclusive, filter_tensors=filter_tensors
            )
            for tid in reversed(self.previous_held_tokens):
                previous_tree = {tid: previous_tree}

            def merge_trees(n1, n2):
                merged = copy(n1)
                for tid, subtree in n2.items():
                    if tid in merged:
                        if tid is None:
                            # this is the only way we can have multiple valid paths to the same byte
                            assert self.previous_sbpe is not None
                            merged[tid] = torch.cat((merged[tid], subtree)).unique()
                            # merged[tid] = torch.from_numpy(
                            #     np.union1d(merged[tid], subtree)
                            # )
                        else:
                            merged[tid] = merge_trees(merged[tid], subtree)
                    else:
                        merged[tid] = subtree

                return merged

            return merge_trees(current_tree, previous_tree)

    def get_streaming_char_pretok(self):
        assert not self.has_digit3_right_just, "Cannot support streaming with right aligned digit groups!"
        return self.StreamingCharPretok(self)

    class StreamingBytePretok:
        def __init__(self, tcs: "ByteConditioning"):
            self.scp = tcs.get_streaming_char_pretok()
            self.buf = []

        def push(self, byte):
            self.buf.append(byte)
            try:
                char = bytes(self.buf).decode()
                self.buf = []
                return self.scp.push(char)

            except UnicodeDecodeError:
                return []

        def split(self):
            assert not self.buf
            return self.scp.split()

        def eval_tree(self, inclusive=False, suffix=b"", filter_tensors=True):
            return self.scp.eval_tree(
                suffix=bytes(self.buf) + suffix,
                inclusive=inclusive,
                filter_tensors=filter_tensors,
            )

    def get_streaming_byte_pretok(self):
        return self.StreamingBytePretok(self)

    def streaming_bpe_open(self, text: Union[str, bytes], inclusive=False, suffix=b"", filter_tensors=True):
        if isinstance(text, str):
            S = self.get_streaming_char_pretok()
        else:
            S = self.get_streaming_byte_pretok()

        trunk = []
        for atom in text:
            trunk.extend(S.push(atom))
        return trunk, S.eval_tree(inclusive=inclusive, suffix=suffix, filter_tensors=True)

    class BytewiseBatchSampler(BaseBytewiseBatchSampler):
        def __init__(
            self,
            tcs: "ByteConditioning",
            batch_size=1,
            filter_tensors=False,
            stop_override=None,
        ):
            self.tcs = tcs
            self.rcm = RadixCacheManager(self.tcs.model, self.tcs.tokenizer)
            self.tic = tcs.token_index_cache
            self.batch_size = batch_size
            self.sbps = [tcs.get_streaming_byte_pretok() for _ in range(batch_size)]
            self.trunks = [[self.tcs.bos] for _ in range(batch_size)]
            self.lens = [0 for _ in range(batch_size)]
            self.trunk_lens = [0 for _ in range(batch_size)]
            self.total_dist_time = 0
            self.filter_tensors = filter_tensors
            self.stop_tokens = (
                stop_override
                if stop_override is not None
                else torch.tensor(
                    [tid for tid, at in tcs.tokenizer.added_tokens_decoder.items() if at.special == True],
                    device=tcs.device,
                )
            )

        def add_context(self, prompts: list[Union[str, bytes]]):
            assert len(prompts) == self.batch_size
            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    if self.tcs.btok.normalizer is not None:
                        prompt = self.tcs.btok.normalizer.normalize_str(prompt)
                    prompt = prompt.encode()

                self.lens[i] += len(prompt)
                for b in prompt:
                    new_tokens = self.sbps[i].push(b)
                    for tid in new_tokens:
                        self.trunk_lens[i] += len(self.tcs.vrev.get(tid, 0))
                    self.trunks[i].extend(new_tokens)

        def add_special_context(self, prompts: list[list[int]]):
            assert len(prompts) == self.batch_size
            for i, prompt in enumerate(prompts):
                if prompt:
                    self.trunks[i].extend(self.sbps[i].split())
                    self.trunks[i].extend(prompt)
                    self.trunk_lens[i] = 0
                    self.lens[i] = 0

        def get_dists(self):
            all_branches = [sbp.eval_tree(inclusive=True, filter_tensors=self.filter_tensors) for sbp in self.sbps]
            results = self.rcm.query([*zip(self.trunks, all_branches)], skip_trunk_logprobs=True, do_gc=False)
            dists = []
            start = time.perf_counter()
            for i, (branches, result) in enumerate(zip(all_branches, results)):
                _, logprob_tree = result
                byte_logprobs = []
                stop_probs = []

                def extract_bytes(eval_tree, lp_tree, past_bytes=0):
                    nonlocal stop_probs
                    for tid, lp_subtree in lp_tree.items():
                        eval_subtree = eval_tree[tid]
                        if tid is None:
                            subset = eval_subtree
                            idx = self.lens[i] - past_bytes
                            if idx == 0:
                                stop_probs.append(torch.logsumexp(lp_subtree[self.stop_tokens], 0))
                            # if idx > 0:
                            #     # the previous byte should be fixed by the prompt
                            #     assert len(self.tic.get(idx - 1)[subset].unique()) == 1
                            selectors, lp_subset = (
                                self.tic.get(idx)[subset],
                                lp_subtree[subset],
                            )
                            byte_logprobs.append(scatter_logsumexp(lp_subset, selectors, dim_size=257))

                        else:
                            extract_bytes(
                                eval_subtree,
                                lp_subtree,
                                past_bytes + len(self.tcs.vrev.get(tid, b"")),
                            )

                extract_bytes(branches, logprob_tree, self.trunk_lens[i])
                stop_prob = torch.logsumexp(torch.tensor(stop_probs, device=self.tcs.device), 0)
                dists.append(
                    torch.hstack(
                        [
                            torch.logsumexp(torch.vstack(byte_logprobs)[:, :-1], 0),
                            stop_prob,
                        ]
                    )
                )

            self.total_dist_time += time.perf_counter() - start
            return torch.vstack(dists)

    def get_bytewise_sampler(self, batch_size=1):
        return self.BytewiseBatchSampler(self, batch_size=batch_size)


