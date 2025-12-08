import time
import warnings
from copy import copy
from dataclasses import dataclass
from typing import Optional, Self, Union

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .utils import logprobs_from_logits


class RadixCacheManager:
    @dataclass
    class CachedToken:
        tid: Optional[int]
        index: Optional[int]
        pos: Optional[int]
        parent: Optional[Self]
        logprobs: Optional[torch.Tensor]
        children: dict[int, Self]
        gc_gen: int

        def __str__(self):
            return f"CT({self.tid} @ {self.pos}, gen{self.gc_gen})"

        __repr__ = __str__

    class SequenceCache:
        seq: list["RadixCacheManager.CachedToken"]
        root: "RadixCacheManager.CachedToken"

        def __init__(self):
            self.seq = []
            self.root = RadixCacheManager.CachedToken(None, None, -1, None, None, {}, 1)

    def __init__(self, model, tokenizer, warn_on_resurrection=False):
        self.model = model
        self.tokenizer = tokenizer
        self.warn_on_resurrection = warn_on_resurrection

        # state
        self.cache = DynamicCache()
        self.cache_meta = None
        self.gc_gen = 0

        # metrics
        self.total_request_time = 0
        self.total_model_time = 0
        self.total_tensor_time = 0
        self.uncached_tokens = 0

    def _make_pad_token(self, index: int, seq_cache: SequenceCache):
        return self.CachedToken(
            self.tokenizer.pad_token_type_id,
            index,
            self.model.config.max_position_embeddings - 1,
            seq_cache.root,
            None,
            {},
            0,  # Never save this token during GC
        )

    def run_gc(self):
        selector = [
            [
                i
                for i, cached_token in enumerate(seq_cache.seq)
                if cached_token.gc_gen == self.gc_gen
            ]
            for seq_cache in self.cache_meta
        ]
        new_cache_size = max(map(len, selector))
        new_pad_tokens = []
        for select in selector:
            new_pads = new_cache_size - len(select)
            new_pad_tokens.append(new_pads)
            select.extend([0] * new_pads)

        selector_pt = torch.tensor(
            selector, device=self.model.device, dtype=torch.long
        )[:, None, :, None]

        def select_kv(layer_tensor):
            nonlocal selector_pt
            selector_pt = selector_pt.to(layer_tensor.device)
            new_shape = list(layer_tensor.shape)
            new_shape[2] = selector_pt.shape[2]
            return torch.gather(
                layer_tensor, 2, selector_pt.expand(new_shape)
            )

        for layer in self.cache.layers:
            layer.keys = select_kv(layer.keys)
            layer.values = select_kv(layer.values)

        # now update the metadata
        for i, (cache, select) in enumerate(zip(self.cache_meta, selector)):
            new_seq = []
            for k, j in enumerate(select):
                if k < new_cache_size - new_pad_tokens[i]:
                    cached_token = cache.seq[j]
                    new_seq.append(cached_token)
                    cached_token.index = k
                    # note: the below code filters the children but we
                    # skip this because we want to be able to detect
                    # resurrected cache entries.

                    # cached_token.children = {
                    #     tid: child
                    #     for tid, child in cached_token.children.items()
                    #     if child.gc_gen == self.gc_gen
                    # }
                else:
                    new_seq.append(self._make_pad_token(k, cache))

            cache.seq = new_seq

            for j, cache_tok in enumerate(cache.seq):
                assert cache_tok.index == j

    def query(
        self,
        batch: list[Union[dict, tuple[list[int], dict]]],
        skip_trunk_logprobs=False,
        do_gc=False,
        logprob_transforms=None,
    ):
        # batch is a list of trees, or (trunk, branches) tuples
        request_start = time.perf_counter()
        batch_size = len(batch)
        self.gc_gen += 1

        # initialize the cache_mapping from the batch_size
        if self.cache_meta is None:
            assert self.gc_gen == 1
            self.cache_meta = [self.SequenceCache() for _ in range(batch_size)]

        assert len(self.cache_meta) == batch_size, "cannot change batch size"

        # check that the cache has the expected size
        ncached = len(self.cache_meta[0].seq)
        assert self.cache.get_seq_length() == ncached, "cache had wrong size!"

        # linearize the eval trees
        all_new_tokens, all_token_backrefs = [], []
        for cache, tree in zip(self.cache_meta, batch):
            # for backwards compatibility with the no-trunk format
            if isinstance(tree, dict):
                tree = ([], tree)

            trunk, branches = tree
            new_tokens = []

            # walk the entire tree
            def linearize_tree(node, cache_node):
                backref = {}
                for tid, subtree in node.items():
                    if tid is None:
                        # None is a request to compute logprobs
                        continue

                    if (
                        (subcache := cache_node.children.get(tid)) is not None
                        and subcache.index < len(cache.seq)
                        and cache.seq[subcache.index] is subcache
                    ):
                        # use the existing cached token
                        # we touched this token so update its gc gen
                        subcache.gc_gen = self.gc_gen

                    else:
                        # if all that's left is the tombstone, then maybe warn
                        if self.warn_on_resurrection and subcache is not None:
                            tok_seq = [subcache.tid]
                            cache_pointer = subcache
                            while (parent := cache_pointer.parent) is not None:
                                tok_seq.append(parent.tid)
                                cache_pointer = parent
                            tok_seq = tok_seq[::-1]
                            warnings.warn(
                                f"Found resurrected token {subcache}: {tok_seq}"
                            )

                        # add a new token to the cache
                        subcache = self.CachedToken(
                            tid,
                            ncached + len(new_tokens),
                            cache_node.pos + 1,
                            cache_node,
                            None,
                            {},
                            self.gc_gen,
                        )
                        new_tokens.append(subcache)
                        cache_node.children[tid] = subcache

                    backref[tid] = linearize_tree(subtree, subcache)

                # if the token is not the root, store a reference to
                # its position so we can lookup its logprobs later
                if cache_node.index is not None:
                    backref[None] = cache_node.index

                return backref

            # replay the trunk back onto the branches
            full_tree = branches
            for tid in reversed(trunk):
                full_tree = {tid: full_tree}

            all_token_backrefs.append(linearize_tree(full_tree, cache.root))
            all_new_tokens.append(new_tokens)

        # pad the new tokens
        maxnew = max(map(len, all_new_tokens))
        if maxnew == 0:
            warnings.warn("wasted a token!")
            maxnew = 1
        for cache, new_tokens in zip(self.cache_meta, all_new_tokens):
            while len(new_tokens) < maxnew:
                new_tokens.append(
                    self._make_pad_token(ncached + len(new_tokens), cache)
                )

        # build the tensors
        input_ids = torch.tensor(
            [[nt.tid for nt in new_tokens] for new_tokens in all_new_tokens],
            device=self.model.device,
            dtype=torch.long,
        )

        position_ids = torch.tensor(
            [[nt.pos for nt in new_tokens] for new_tokens in all_new_tokens],
            device=self.model.device,
            dtype=torch.long,
        )

        attention_mask = torch.full(
            (
                batch_size,
                1,
                maxnew,
                ncached + maxnew,
            ),
            -torch.inf,
            dtype=self.model.dtype,
            device=self.model.device,
        )

        batch_idxs, new_idxs, past_idxs = [], [], []
        for bi, (cache, new_tokens) in enumerate(zip(self.cache_meta, all_new_tokens)):
            for ni, nt in enumerate(new_tokens):
                while True:
                    pi = nt.index
                    batch_idxs.append(bi)
                    new_idxs.append(ni)
                    past_idxs.append(pi)
                    if nt.parent is cache.root:
                        break
                    nt = nt.parent

        attention_mask[batch_idxs, 0, new_idxs, past_idxs] = 0

        # call the model
        model_start = time.perf_counter()
        fwd = self.model.forward(
            input_ids,
            use_cache=True,
            past_key_values=self.cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        self.cache = fwd.past_key_values
        logprobs = F.log_softmax(fwd.logits.to(torch.float32), -1)
        self.total_model_time += time.perf_counter() - model_start
        self.uncached_tokens += input_ids.shape[-1]

        # roll the new tokens into the cache
        for new_tokens, lp_slice in zip(all_new_tokens, logprobs):
            for nt, lps in zip(new_tokens, lp_slice):
                nt.logprobs = lps

        for cache, new_tokens in zip(self.cache_meta, all_new_tokens):
            cache.seq.extend(new_tokens)
            assert len(cache.seq) == ncached + maxnew

        for new_tokens in all_new_tokens:
            for tok in new_tokens:
                assert tok.index < ncached + maxnew

        # optionally apply any transformations to the logprobs
        def transform_logprobs(lp):
            if logprob_transforms is not None:
                return logprobs_from_logits(lp, **logprob_transforms)
            return lp

        # pull the logprobs back into the tree using the backrefs
        def lookup_backrefs(cache_seq, tree, backrefs, cum_logprob=0):
            if not isinstance(tree, dict):
                # pull the trunk out as a list of logprobs if it was passed
                (trunk, branches), bpointer, trunk_logprobs = tree, backrefs, []
                for tid in trunk:
                    if (
                        bindex := bpointer.get(None)
                    ) is not None and not skip_trunk_logprobs:
                        # the first token has no loss
                        trunk_logprobs.append(
                            transform_logprobs(cache_seq[bindex].logprobs)[tid]
                        )
                    bpointer = bpointer[tid]

                return trunk_logprobs, lookup_backrefs(cache_seq, branches, bpointer, 0)

            result = {}
            for tid, subtree in tree.items():
                if tid is None:
                    # the logprobs are requested here
                    result[tid] = (
                        transform_logprobs(cache_seq[backrefs[None]].logprobs)
                        + cum_logprob
                    )

                else:
                    result[tid] = lookup_backrefs(
                        cache_seq,
                        subtree,
                        backrefs[tid],
                        cum_logprob
                        + (
                            transform_logprobs(cache_seq[bindex].logprobs)[tid]
                            if (bindex := backrefs.get(None)) is not None
                            else 0
                        ),
                    )

            return result

        tensor_start = time.perf_counter()
        result = [
            lookup_backrefs(cache.seq, tree, new_token_backrefs)
            for cache, tree, new_token_backrefs in zip(
                self.cache_meta, batch, all_token_backrefs
            )
        ]

        # optionally, run the copying garbage collector
        if do_gc:
            self.run_gc()

        self.total_tensor_time += time.perf_counter() - tensor_start
        self.total_request_time += time.perf_counter() - request_start
        return result

    def export_cache(self, batch: list[list[int]]):
        selector = []
        for cm, seq in zip(self.cache_meta, batch):
            sel, pointer = [], cm.root
            for tid in seq[:-1]:
                pointer = pointer.children[tid]
                sel.append(pointer.pos)
            selector.append(sel)

        new_cache_size = max(map(len, selector))

        new_pad_tokens = []
        for i, select in enumerate(selector):
            new_pads = new_cache_size - len(select)
            new_pad_tokens.append(new_pads)
            selector[i] = [0] * new_pads + select

        selector_pt = torch.tensor(
            selector, device=self.model.device, dtype=torch.long
        )[:, None, :, None]

        def select_cache(layer_tensor):
            new_shape = list(layer_tensor.shape)
            new_shape[2] = selector_pt.shape[2]
            return torch.gather(layer_tensor, 2, selector_pt.expand(new_shape))

        new_cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(
            zip(self.cache.key_cache, self.cache.value_cache)
        ):
            new_cache.update(select_cache(key), select_cache(value), layer_idx)

        return new_cache
