from __future__ import annotations

import heapq
import itertools
import os
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Self, Tuple

import torch
import torch.nn.functional as F


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


def logprobs_from_logits(
    logits: torch.Tensor,
    temperature: float = 1,
    top_k: int | None = None,
    top_p: float | None = None,
    filter_value: float = -float("Inf"),
):
    # Adapted from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    logits = logits.detach().clone()

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        top_k = min(top_k, logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        )
        logits[indices_to_remove] += filter_value

    if top_p is not None:
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] += filter_value
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))

    logits_max = logits.max(dim=-1, keepdim=True).values
    scaled = (logits - logits_max) / temperature
    logprobs = torch.log_softmax(scaled, dim=-1)
    return logprobs


def sample_from_logits(
    logits: torch.Tensor,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    filter_value: float = -float("Inf"),
    generator: torch.Generator | None = None,
):
    if not do_sample or temperature < 1e-4:
        return torch.argmax(logits, dim=-1)

    logprobs = logprobs_from_logits(
        logits=logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        filter_value=filter_value,
    )

    return torch.multinomial(torch.softmax(logprobs, dim=-1), 1, generator=generator)[
        ..., 0
    ]


def sample_from_prob_tree(
    batch_tree,
    prob_tree,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
):
    trunk, branches = prob_tree
    btpointer = batch_tree

    # Fast forward the batch tree to match branches
    for tid, _ in trunk:
        btpointer = btpointer[tid]

    # Flatten the tree into a logprob tensor
    def collect_prob_nodes(bt, pt, past=[]):
        result = []
        for k, v in pt.items():
            if k is None:
                result.append((v, bt[k], copy(past)))
            else:
                past.append(k)
                result.extend(collect_prob_nodes(bt[k], v, past))
                past.pop()
        return result

    nodes = collect_prob_nodes(btpointer, branches)
    probs = torch.hstack([ps[tids] for ps, tids, _ in nodes]) / temperature
    logprobs = torch.log_softmax(probs, 0)
    idx = sample_from_logits(
        logprobs,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    ).item()

    # Translate the index into a context
    result = [tid for tid, _ in trunk]
    for ps, ids, past in nodes:
        if idx < len(ids):
            result.extend(past)
            result.append(ids[idx].item())
            return result
        idx -= len(ids)


def scatter_logsumexp(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim_size: int | None = None,
    safe_all_neg_inf: bool = False,
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
    m = src.new_full((dim_size,), -torch.inf)
    m.scatter_reduce_(0, index, src.detach(), reduce="amax", include_self=False)
    if not safe_all_neg_inf:
        # handle the all-(-inf) case
        m = torch.nan_to_num(m, nan=None, neginf=0, out=None if m.requires_grad else m)
        # 2. exponentiate shifted values and sum per bucket
        shifted_exp = (src - m[index]).exp()
        s = torch.zeros_like(m).scatter_add_(0, index, shifted_exp)
        # 3. log-sum-exp
        return m + s.log()

    all_neg_inf = torch.isneginf(m)
    m = torch.where(all_neg_inf, torch.zeros_like(m), m)
    # 2. exponentiate shifted values and sum per bucket
    shifted_exp = (src - m[index]).exp()
    s = torch.zeros_like(m).scatter_add_(0, index, shifted_exp)
    s = torch.where(all_neg_inf, torch.ones_like(s), s)
    # 3. log-sum-exp
    out = m + s.log()
    return torch.where(all_neg_inf, m.new_full(out.shape, -torch.inf), out)


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


class RingDeque:
    """Deque via circular buffer:
    - Amortized O(1) append/appendleft/pop/popleft
    - __getitem__(int) in O(1); __getitem__(slice) → generator over the slice
    - Capacity doubles when full
    """

    __slots__ = ("_data", "_head", "_tail", "_size")

    def __init__(self, iterable=None, initial_capacity=8):
        cap = 1
        base = max(1, int(initial_capacity))
        while cap < base:
            cap <<= 1
        self._data = [None] * cap
        self._head = 0  # index of logical element 0
        self._tail = 0  # one past the last element
        self._size = 0
        if iterable:
            for x in iterable:
                self.append(x)

    # ---------- helpers ----------
    def _phys_index(self, i):
        """Map logical index i (0..size-1, accepts negative) to physical slot."""
        n = self._size
        if not -n <= i < n:
            raise IndexError("deque index out of range")
        if i < 0:
            i += n
        cap = len(self._data)
        return (self._head + i) & (cap - 1)

    def _ensure_capacity(self, needed):
        if needed <= len(self._data):
            return
        old_cap = len(self._data)
        new_cap = old_cap * 2 if old_cap else 8
        while new_cap < needed:
            new_cap *= 2
        new_data = [None] * new_cap

        # Copy existing elements in logical order into new_data[0:size]
        if self._size:
            if self._head < self._tail:
                new_data[0 : self._size] = self._data[self._head : self._tail]
            else:
                left = old_cap - self._head
                new_data[0:left] = self._data[self._head :]
                new_data[left : left + self._tail] = self._data[: self._tail]

        self._data = new_data
        self._head = 0
        self._tail = self._size  # one past last

    # ---------- core API ----------
    def append(self, value):
        self._ensure_capacity(self._size + 1)
        cap = len(self._data)
        mask = cap - 1
        self._data[self._tail] = value
        self._tail = (self._tail + 1) & mask
        self._size += 1

    def appendleft(self, value):
        self._ensure_capacity(self._size + 1)
        cap = len(self._data)
        mask = cap - 1
        self._head = (self._head - 1) & mask
        self._data[self._head] = value
        self._size += 1

    def pop(self):
        if not self._size:
            raise IndexError("pop from an empty deque")
        cap = len(self._data)
        mask = cap - 1
        self._tail = (self._tail - 1) & mask
        val = self._data[self._tail]
        self._data[self._tail] = None
        self._size -= 1
        return val

    def popleft(self):
        if not self._size:
            raise IndexError("popleft from an empty deque")
        cap = len(self._data)
        mask = cap - 1
        val = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) & mask
        self._size -= 1
        return val

    # ---------- peeks ----------
    def peek(self):
        """Right-end (tail-1) element without removing it."""
        if not self._size:
            raise IndexError("peek from an empty deque")
        cap = len(self._data)
        mask = cap - 1
        return self._data[(self._tail - 1) & mask]

    def peekleft(self):
        """Left-end (head) element without removing it."""
        if not self._size:
            raise IndexError("peekleft from an empty deque")
        return self._data[self._head]

    # ---------- Python protocol ----------
    def __len__(self):
        return self._size

    def __iter__(self):
        cap = len(self._data)
        mask = cap - 1
        for i in range(self._size):
            yield self._data[(self._head + i) & mask]

    def __repr__(self):
        return f"RingDeque([{', '.join(repr(x) for x in self)}])"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._size)
            if step == 0:
                raise ValueError("slice step cannot be zero")
            # Capture current storage to keep iteration stable.
            data = self._data
            head = self._head
            cap = len(data)
            mask = cap - 1

            def _gen(s=start, e=stop, st=step, d=data, h=head, m=mask):
                for i in range(s, e, st):
                    yield d[(h + i) & m]

            return _gen()
        else:
            return self._data[self._phys_index(key)]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise TypeError("slice assignment not supported")
        self._data[self._phys_index(key)] = value
