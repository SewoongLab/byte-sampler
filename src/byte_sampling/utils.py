from __future__ import annotations

import heapq
import itertools
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import io
import gzip
import simdjson as json
import torch

import tqdm.auto as tqdm


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def is_valid_unicode(data):
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def build_trie(seq):
    trie = {}
    for tok in seq:
        node = trie
        for i, b in enumerate(tok):
            node = node.setdefault(b, {})
        node[None] = True
    return trie


def trie_lookup(trie, key):
    node = trie
    for b in key:
        if b not in node:
            return False
        node = node[b]
    return node.get(None, False)

def walk_trie(trie):
    results = []
    if trie.get(None):
        results.append([])

    for b, subtrie in trie.items():
        if b is not None:
            for rest in walk_trie(subtrie):
                results.append([b] + rest)

    return results

def bytes_to_unicode():
    """
    MJ: STOLEN DIRECTLY FROM https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    --------------
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class PriorityQueue:
    def __init__(self, items=None, max_queue=True):
        self.pq = []
        self.removed = object()
        self.entry_finder = {}
        self.counter = itertools.count()
        self.max_queue = max_queue
        if items is not None:
            for el, priority in items:
                if self.max_queue:
                    priority = -priority
                assert el not in self
                count = next(self.counter)
                entry = [priority, count, el]
                self.entry_finder[el] = entry
                self.pq.append(entry)
            heapq.heapify(self.pq)

    def add(self, el, priority):
        if self.max_queue:
            priority = -priority
        if el in self:
            self.remove(el)
        count = next(self.counter)
        entry = [priority, count, el]
        self.entry_finder[el] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, el):
        entry = self.entry_finder.pop(el)
        entry[-1] = self.removed

    def pop(self):
        while self.pq:
            priority, count, el = heapq.heappop(self.pq)
            if el is not self.removed:
                del self.entry_finder[el]
                if self.max_queue:
                    priority = -priority
                return el, priority
        raise KeyError("pop from an empty priority queue")

    def peek(self):
        while self.pq:
            priority, count, el = self.pq[0]
            if el is self.removed:
                heapq.heappop(self.pq)
                continue

            if self.max_queue:
                priority = -priority
            return el, priority
        raise KeyError("peek from an empty priority queue")

    def lookup(self, el, default=None):
        priority = self.entry_finder.get(el, (default,))[0]
        if self.max_queue:
            priority = -priority
        return priority

    def __getitem__(self, el):
        return self.entry_finder[el][0]

    def __contains__(self, el):
        return el in self.entry_finder

    def __len__(self):
        return len(self.entry_finder)


def get_file_len(f):
    cur = f.tell()
    f.seek(0, os.SEEK_END)
    out = f.tell()
    f.seek(cur)
    return out


def sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    filter_value: float = -float("Inf"),
    generator: torch.Generator | None = None,
):
    # Adapted from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    if not do_sample or temperature < 1e-4:
        return torch.argmax(logits, dim=-1)

    logits = logits.detach().clone()

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        )
        logits[indices_to_remove] = filter_value

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = filter_value
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))

    scaled = (logits - logits.max()) / temperature
    probs = torch.softmax(scaled, dim=-1)

    return torch.multinomial(probs, 1, generator=generator)[..., 0]


def read_json(path: PathOrStr) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def seed_all(seed: int):
    import random
    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)


def scatter_logsumexp(
    src: torch.Tensor, index: torch.Tensor, *, dim_size: int | None = None
) -> torch.Tensor:
    """
    Numerically-stable grouped log-sum-exp.
    Parameters
    ----------
    src      : 1-D float tensor (values to reduce)
    index    : 1-D int64 tensor, same length as `src`
    dim_size : number of buckets; default = index.max() + 1
    Returns
    -------
    out : tensor with shape (dim_size,)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    # 1. per-bucket max for numerical stability
    m = torch.full((dim_size,), -torch.inf, device=src.device)
    m.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    # 2. exponentiate shifted values and sum per bucket
    shifted_exp = (src - m[index]).exp()
    s = torch.zeros_like(m).scatter_add_(0, index, shifted_exp)
    # 3. log-sum-exp
    return m + s.log()


class DoublyLinkedList:
    @dataclass
    class Node:
        obj: object
        p: Optional[Self]
        n: Optional[Self]

        def __str__(self):
            return f"Node({self.obj})"

        __repr__ = __str__

    def __init__(self, lst):
        self.head = self.Node(lst[0], None, None)
        node = self.head
        for i, obj in enumerate(lst):
            if i == 0:
                continue
            newnode = self.Node(obj, node, None)
            node.n = newnode
            node = newnode
        self.tail = node

    def __iter__(self):
        def inner():
            node = self.head
            while True:
                yield node
                if not (node := node.n):
                    break

        return inner()

    def __str__(self):
        items = [node.obj for node in self]
        return f"<{str(items)[1:-1]}>"

    __repr__ = __str__
