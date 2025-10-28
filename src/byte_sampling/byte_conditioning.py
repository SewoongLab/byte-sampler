import bisect
import heapq
import itertools as it
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from typing import Optional, Self, Union

import numpy as np
import regex
import simdjson as json
import torch
from ahocorasick_rs import AhoCorasick, MatchKind
from transformers import AutoModelForCausalLM, AutoTokenizer

from .radix_cache import RadixCacheManager
from .streaming_added_tokens import StreamingAddedTokens
from .streaming_bpe import StreamingBPE
from .streaming_pretok import StreamingCharPretok
from .utils import (DoublyLinkedList, build_trie, bytes_to_unicode,
                    scatter_logsumexp)


class BaseBytewiseBatchSampler(ABC):
    @abstractmethod
    def add_context(self, prompts: list[Union[str, bytes]]):
        pass

    @abstractmethod
    def add_special_context(self, prompts: list[list[int]]):
        pass

    @abstractmethod
    def get_dists(self, **kwargs) -> torch.Tensor:
        pass


class ByteConditioning(object):
    class TokenIndexerCache:
        """Class to cache the arrays of nth bytes of each token"""

        def __init__(self, tcs):
            self.tcs = tcs
            self.device = tcs.model.device if tcs.model else "cpu"
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
                dict(device_map="auto", torch_dtype=torch.bfloat16)
                if load_kwargs is None
                else load_kwargs
            )
            if not skip_model:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_or_dir, **load_kwargs
                )
            else:
                self.model = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_or_dir)

        self.device = (
            self.model.device if self.model is not None else torch.get_default_device()
        )
        self.dtype = (
            self.model.dtype if self.model is not None else torch.get_default_dtype()
        )

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
        # for tok in added_tokens:
        #     assert not tok.get("single_word"), f"{tok}"
        #     assert not tok.get("lstrip"), f"{tok}"
        #     assert not tok.get("rstrip"), f"{tok}"

        self.split_trie = AhoCorasick(
            [tok["content"] for tok in added_tokens if not tok["normalized"]],
            matchkind=MatchKind.LeftmostLongest,
        )
        self.split_ids = [tok["id"] for tok in added_tokens if not tok["normalized"]]

        self.split_normalized_trie = AhoCorasick(
            [tok["content"] for tok in added_tokens if tok["normalized"]],
            matchkind=MatchKind.LeftmostLongest,
        )
        self.split_normalized_ids = [
            tok["id"] for tok in added_tokens if tok["normalized"]
        ]
        # if self.btok.normalizer is not None:
        #     assert not self.split_normalized_ids

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
            ll = DoublyLinkedList(
                [(i, self.vocab[bytes([b])]) for i, b in enumerate(buffer)]
            )
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

                ab = merge_map[(a, b)]
                if ab in resolve_map:
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
        for tbytes, tid in sorted(
            self.vocab.items(), key=lambda pair: len(pair[0]), reverse=True
        ):
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

    def _walk_vtrie(self, vtrie, skip_unreachable=True):
        if None in vtrie:
            tid = vtrie[None]
            if not skip_unreachable or tid not in self.unreachable_tokens:
                yield tid

        for b, subtrie in vtrie.items():
            if b is not None:
                yield from self._walk_vtrie(subtrie)

    def _get_walk_cached(self, prefix: bytes, skip_unreachable=True):
        cache_key = prefix, skip_unreachable
        if cache_key in self.walk_cache:
            return self.walk_cache[cache_key]
        pointer = self.vtrie
        for b in prefix:
            if b not in pointer:
                return np.array([], int)
            pointer = pointer[b]
        candidates = np.fromiter(
            self._walk_vtrie(pointer, skip_unreachable=skip_unreachable), int
        )
        if len(candidates) > self.walk_cache_min_size:
            self.walk_cache[cache_key] = candidates
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

    def _valid_r_filtered(self, l: int | None, prefix: bytes) -> torch.Tensor:
        candidates = self._get_walk_cached(prefix, skip_unreachable=l is not None)

        if l is None:
            result = candidates
        elif l in self.unreachable_tokens:
            result = np.array([], dtype=int)
        else:
            invalid = np.fromiter(self._invalid_r_filtered(l, prefix), int)
            result = np.setdiff1d(candidates, invalid, assume_unique=True)

        return torch.from_numpy(result).to(device=self.device)

    def _valid_r_unfiltered(self, prefix: Iterable[int]) -> torch.Tensor:
        mask = torch.ones(
            self.tokenizer.vocab_size,
            dtype=torch.bool,
            device=self.device,
        )

        for i, b in enumerate(prefix):
            mask &= self.token_index_cache.get(i) == b

        valid_tokens = torch.arange(self.tokenizer.vocab_size, device=self.device)[mask]

        return valid_tokens

    def get_streaming_bpe(self):
        return StreamingBPE(self)

    def get_streaming_char_pretok(self):
        return StreamingCharPretok(self)

    def get_streaming_added_tokens(self):
        return StreamingAddedTokens(self)

    class StreamingByteTree:
        def __init__(self, tcs: "ByteConditioning"):
            self.sat = tcs.get_streaming_added_tokens()
            self.buf = []

        def push(self, byte):
            self.buf.append(byte)
            try:
                char = bytes(self.buf).decode()
                self.buf = []
                return self.sat.push(char)

            except UnicodeDecodeError:
                return []

        def split(self):
            assert not self.buf
            return self.sat.split()

        def eval_tree(self, inclusive=False, suffix=b"", filter_tensors=True):
            return self.sat.eval_tree(
                suffix=bytes(self.buf) + suffix,
                inclusive=inclusive,
                filter_tensors=filter_tensors,
            )

    def get_streaming_byte_tree(self):
        return self.StreamingByteTree(self)

    def streaming_bpe_open(
        self, text: Union[str, bytes], inclusive=False, suffix=b"", filter_tensors=True
    ):
        if isinstance(text, str):
            text = text.encode()

        S = self.get_streaming_byte_tree()

        trunk = []
        for atom in text:
            trunk.extend(S.push(atom))
        return trunk, S.eval_tree(
            inclusive=inclusive, suffix=suffix, filter_tensors=True
        )

    class BytewiseBatchSampler(BaseBytewiseBatchSampler):
        def __init__(
            self,
            bc: "ByteConditioning",
            batch_size=1,
            filter_tensors=False,
            do_gc=True,
            stop_override=None,
        ):
            self.bc = bc
            self.rcm = RadixCacheManager(
                self.bc.model, self.bc.tokenizer, warn_on_resurrection=True
            )
            self.tic = bc.token_index_cache
            self.batch_size = batch_size
            self.sbps = [bc.get_streaming_byte_tree() for _ in range(batch_size)]
            self.trunks = [[self.bc.bos] for _ in range(batch_size)]
            self.lens = [0 for _ in range(batch_size)]
            self.trunk_lens = [0 for _ in range(batch_size)]
            self.total_dist_time = 0
            self.filter_tensors = filter_tensors
            self.do_gc = do_gc
            self.stop_tokens = (
                stop_override
                if stop_override is not None
                else torch.tensor(
                    [
                        tid
                        for tid, at in bc.tokenizer.added_tokens_decoder.items()
                        if at.special
                    ],
                    device=bc.device,
                )
            )

        def add_context(self, prompts: list[Union[str, bytes]]):
            assert len(prompts) == self.batch_size
            for i, prompt in enumerate(prompts):
                if isinstance(prompt, str):
                    if self.bc.btok.normalizer is not None:
                        prompt = self.bc.btok.normalizer.normalize_str(prompt)
                    prompt = prompt.encode()

                self.lens[i] += len(prompt)
                for b in prompt:
                    new_tokens = self.sbps[i].push(b)
                    for tid in new_tokens:
                        self.trunk_lens[i] += len(self.bc.vrev.get(tid, 0))
                    self.trunks[i].extend(new_tokens)

        def add_special_context(self, prompts: list[list[int]]):
            assert len(prompts) == self.batch_size
            for i, prompt in enumerate(prompts):
                if prompt:
                    self.trunks[i].extend(self.sbps[i].split())
                    self.trunks[i].extend(prompt)
                    self.trunk_lens[i] = 0
                    self.lens[i] = 0

        def tree_inference(
            self,
            *,
            inclusive=True,
            filter_tensors=None,
            do_gc=None,
            logprob_transforms=None,
        ):
            if filter_tensors is None:
                filter_tensors = self.filter_tensors
            if do_gc is None:
                do_gc = self.do_gc

            # compute what token probabilities are needed
            all_branches = [
                sbp.eval_tree(inclusive=inclusive, filter_tensors=filter_tensors)
                for sbp in self.sbps
            ]

            # execute the token-level query
            results = self.rcm.query(
                [*zip(self.trunks, all_branches)],
                skip_trunk_logprobs=True,
                do_gc=do_gc,
                logprob_transforms=logprob_transforms,
            )

            return all_branches, results

        def get_dists(
            self, *, filter_tensors=None, do_gc=None, logprob_transforms=None
        ):
            all_branches, results = self.tree_inference(
                inclusive=True,
                filter_tensors=filter_tensors,
                do_gc=do_gc,
                logprob_transforms=logprob_transforms,
            )

            # aggregate the token-level probabilities into byte-level ones
            dists = []
            start = time.perf_counter()
            for i, (branches, (_, logprob_tree)) in enumerate(
                zip(all_branches, results)
            ):
                byte_logprobs, stop_logprobs = [], []

                # walk the tree
                def extract_bytes(eval_tree, lp_tree, past_bytes=0):
                    for tid, lp_subtree in lp_tree.items():
                        eval_subtree = eval_tree[tid]
                        if tid is None:
                            subset = eval_subtree
                            # how many bytes until the end of the prompt
                            prompt_offset = self.lens[i] - past_bytes

                            if prompt_offset == 0:
                                # only process special tokens at the end of the prompt
                                stop_logprobs.append(
                                    torch.logsumexp(lp_subtree[self.stop_tokens], 0)
                                )

                            # sanity check: the previous byte should be fixed by the prompt
                            # if idx > 0:
                            #     assert len(self.tic.get(idx - 1)[subset].unique()) == 1

                            selectors = self.tic.get(prompt_offset)[subset]
                            lp_subset = lp_subtree[subset]

                            # 257th byte is for tokens with no byte representation
                            # (e.g. special ones) which are handled separately
                            byte_logprobs.append(
                                scatter_logsumexp(lp_subset, selectors, dim_size=257)
                            )

                        else:
                            extract_bytes(
                                eval_subtree,
                                lp_subtree,
                                past_bytes + len(self.bc.vrev.get(tid, b"")),
                            )

                extract_bytes(branches, logprob_tree, self.trunk_lens[i])
                stop_logprob = torch.logsumexp(
                    torch.tensor(stop_logprobs, device=self.bc.device), 0
                )
                dists.append(
                    torch.hstack(
                        [
                            torch.logsumexp(torch.vstack(byte_logprobs)[:, :-1], 0),
                            stop_logprob,
                        ]
                    )
                )

            self.total_dist_time += time.perf_counter() - start
            return torch.vstack(dists)

    def get_bytewise_sampler(self, batch_size=1):
        return self.BytewiseBatchSampler(self, batch_size=batch_size)
