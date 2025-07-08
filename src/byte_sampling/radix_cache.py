import time
from dataclasses import dataclass
from typing import Optional, Self, Union

import torch
import torch.nn.functional as F
from transformers import DynamicCache

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

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = DynamicCache()
        self.cache_meta = None
        self.total_request_time = 0
        self.total_model_time = 0
        self.total_tensor_time = 0
        self.gc_gen = 0

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

    def query(
        self,
        batch: list[Union[dict, tuple[list[int], dict]]],
        skip_trunk_logprobs=False,
        do_gc=False,
    ):
        # batch is a list of trees
        request_start = time.perf_counter()
        bsize = len(batch)
        self.gc_gen += 1

        # initialize the cache_mapping
        if self.cache_meta is None:
            assert self.gc_gen == 1
            self.cache_meta = [self.SequenceCache() for _ in range(bsize)]

        assert len(self.cache_meta) == bsize, f"cannot change batch size"

        # linearize the eval trees
        all_new_tokens, all_token_backrefs = [], []
        ncached = len(self.cache_meta[0].seq)
        assert self.cache.get_seq_length() == ncached
        for cache, tree in zip(self.cache_meta, batch):
            if isinstance(tree, dict):
                tree = ([], tree)

            trunk, branches = tree
            new_tokens = []

            def linearize_tree(node, cache):
                # print(node, cache)
                backref = {}
                for tid, subtree in node.items():
                    if tid is None:
                        continue
                    if tid in cache.children:
                        subcache = cache.children[tid]
                        # We touched this token so update its gc gen
                        subcache.gc_gen = self.gc_gen
                    else:
                        subcache = self.CachedToken(
                            tid,
                            ncached + len(new_tokens),
                            cache.pos + 1,
                            cache,
                            None,
                            {},
                            self.gc_gen,
                        )
                        new_tokens.append(subcache)
                        cache.children[tid] = subcache

                    backref[tid] = linearize_tree(subtree, subcache)

                if cache.index is not None:
                    backref[None] = cache.index

                return backref

            full_tree = branches
            for tid in reversed(trunk):
                full_tree = {tid: full_tree}

            all_token_backrefs.append(linearize_tree(full_tree, cache.root))
            all_new_tokens.append(new_tokens)

        # pad the new tokens
        maxnew = max(map(len, all_new_tokens))
        if maxnew == 0:
            print("wasted a token!")
            maxnew = 1
        for cache, new_tokens in zip(self.cache_meta, all_new_tokens):
            while len(new_tokens) < maxnew:
                new_tokens.append(self._make_pad_token(ncached + len(new_tokens), cache))

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
                bsize,
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

        # roll the new tokens into the cache
        for new_tokens, lp_slice in zip(all_new_tokens, logprobs):
            for nt, lps in zip(new_tokens, lp_slice):
                nt.logprobs = lps

        for cache, new_tokens in zip(self.cache_meta, all_new_tokens):
            cache.seq.extend(new_tokens)

        # pull the logprobs back into the tree using the backrefs
        def lookup_backrefs(cache_seq, tree, backrefs, cum_logprob=0):
            if not isinstance(tree, dict):
                # pull the trunk out as a list of logprobs if it was passed
                (trunk, branches), bpointer, trunk_logprobs = tree, backrefs, []
                for tid in trunk:
                    if None in bpointer and not skip_trunk_logprobs:
                        # the first token has no loss
                        trunk_logprobs.append(cache_seq[bpointer[None]].logprobs[tid])
                    bpointer = bpointer[tid]

                return trunk_logprobs, lookup_backrefs(cache_seq, branches, bpointer, 0)

            result = {}
            for tid, subtree in tree.items():
                if tid is None:
                    result[tid] = cache_seq[backrefs[None]].logprobs + cum_logprob

                else:
                    result[tid] = lookup_backrefs(
                        cache_seq,
                        subtree,
                        backrefs[tid],
                        cum_logprob + (cache_seq[backrefs[None]].logprobs[tid] if None in backrefs else 0),
                    )

            return result

        tensor_start = time.perf_counter()
        result = [
            lookup_backrefs(cache.seq, tree, new_token_backrefs)
            for cache, tree, new_token_backrefs in zip(self.cache_meta, batch, all_token_backrefs)
        ]

        # optionally, run the copying garbage collector
        if do_gc:
            selector = [
                [i for i, cached_token in enumerate(seq_cache.seq) if cached_token.gc_gen == self.gc_gen]
                for seq_cache in self.cache_meta
            ]
            new_cache_size = max(map(len, selector))
            new_pad_tokens = []
            for seq_select in selector:
                new_pads = new_cache_size - len(seq_select)
                new_pad_tokens.append(new_pads)
                seq_select.extend([0] * new_pads)

            selector_pt = torch.tensor(selector, device=self.model.device, dtype=torch.long)[:, None, :, None]
            for cache in (self.cache.key_cache, self.cache.value_cache):
                for i, layer_tensor in enumerate(cache):
                    new_shape = list(layer_tensor.shape)
                    new_shape[2] = selector_pt.shape[2]
                    cache[i] = torch.gather(layer_tensor, 2, selector_pt.expand(new_shape))

            # now update the metadata
            for i, (seq_cache, seq_select) in enumerate(zip(self.cache_meta, selector)):
                new_seq = []
                for k, j in enumerate(seq_select):
                    cached_token = seq_cache.seq[j]
                    new_seq.append(cached_token)
                    cached_token.index = k

                seq_cache.seq = new_seq
                # set the last new_pad_tokens[i] entries to pad tokens
                for j in range(new_cache_size - new_pad_tokens[i], new_cache_size):
                    seq_cache.seq[j] = self._make_pad_token(j, seq_cache)
                pass

        self.total_tensor_time += time.perf_counter() - tensor_start
        self.total_request_time += time.perf_counter() - request_start
        return result
