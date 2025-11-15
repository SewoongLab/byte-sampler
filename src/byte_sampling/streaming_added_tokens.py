import itertools as it
from collections import deque
from dataclasses import dataclass, field
from typing import Self

import torch
from icu import CanonicalIterator, UnicodeString

from .streaming_pretok import StreamingCharPretok
from .utils import RingDeque, build_trie


class StreamingAddedTokens:
    @dataclass(slots=True)
    class State:
        identifier: int
        depth: int
        symbol: str = None
        transitions: dict = field(default_factory=dict)
        parent: Self | None = None
        success: bool = False
        tid: int | None = None
        longest_strict_suffix: Self | None = None
        terminal: bool = True

        def __str__(self):
            transitions_as_string = ",".join(
                [
                    "{0} -> {1}".format(key, value.identifier)
                    for key, value in self.transitions.items()
                ]
            )
            return "State {0}. Transitions: {1}".format(
                self.identifier, transitions_as_string
            )

        __repr__ = __str__

    @dataclass(slots=True)
    class Match:
        l: int
        r: int
        tid: int
        outbuf: list[int]
        scp: StreamingCharPretok
        state: "StreamingAddedTokens.State"

    def __init__(self, tcs, special_tokens=None):
        self.tcs = tcs
        self.base_scp = tcs.get_streaming_char_pretok()
        self.base_idx = 0
        self.normalizer = tcs.tokenizer.backend_tokenizer.normalizer

        self._zero_state = self.State(0, 0)
        self._zero_state.longest_strict_suffix = self._zero_state
        self._counter = 1
        self.state = self._zero_state
        self.idx = 0
        self.chain = deque()
        self.buf = RingDeque(initial_capacity=16)
        self.buf_idx = 0

        self.added = {}

        if not special_tokens:
            for tid, at in self.tcs.tokenizer.added_tokens_decoder.items():
                # if at.special:
                #     continue
                # assert not at.lstrip
                # assert not at.rstrip
                assert not at.single_word
                if self.normalizer is not None:
                    normalized = self.normalizer.normalize_str(at.content)
                    if at.normalized:
                        self.added[tid] = at.content
                    elif self.unique_canon_variant(normalized):
                        self.added[tid] = normalized
                    else:
                        raise NotImplementedError
                else:
                    self.added[tid] = at.content
        else:
            self.added = special_tokens

        self.arev = {v: k for k, v in self.added.items()}
        self.trie = build_trie(self.added.values())
        for tok in self.added.values():
            self._add(tok)

        self._search_lss_for_children(self._zero_state)

    @classmethod
    def unique_canon_variant(cls, s: str) -> set[str]:
        variants = CanonicalIterator(UnicodeString(s))
        buf = list(it.islice(iter(variants), 2))  # read at most two
        return len(buf) == 1

    def _add(self, keyword: str):
        original_keyword = keyword
        if len(keyword) <= 0:
            return
        current_state = self._zero_state
        for char in keyword:
            current_state.terminal = False
            try:
                current_state = current_state.transitions[char]
            except KeyError:
                next_state = self.State(
                    identifier=self._counter,
                    depth=current_state.depth + 1,
                    parent=current_state,
                    symbol=char,
                )
                self._counter += 1
                current_state.transitions[char] = next_state
                current_state = next_state
        current_state.success = True
        current_state.tid = self.arev[original_keyword]

    def _search_lss_for_children(self, zero_state):
        processed = set()
        to_process = [zero_state]
        while to_process:
            state = to_process.pop()
            processed.add(state.identifier)
            for child in state.transitions.values():
                if child.identifier not in processed:
                    self._search_lss(child)
                    to_process.append(child)

    def _search_lss(self, state):
        zero_state = self._zero_state
        parent = state.parent
        traversed = parent.longest_strict_suffix
        while True:
            if (
                state.symbol in traversed.transitions
                and traversed.transitions[state.symbol] is not state
            ):
                state.longest_strict_suffix = traversed.transitions[state.symbol]
                break
            elif traversed is zero_state:
                state.longest_strict_suffix = zero_state
                break
            else:
                traversed = traversed.longest_strict_suffix
        suffix = state.longest_strict_suffix
        if suffix is zero_state:
            return
        if suffix.longest_strict_suffix is None:
            self._search_lss(suffix)
        for symbol, next_state in suffix.transitions.items():
            if symbol not in state.transitions:
                state.transitions[symbol] = next_state

    def _enumerate_matches(self, state):
        while state is not self._zero_state:
            if state.success:
                yield state, self.idx - state.depth

            state = state.longest_strict_suffix

    def _walk_state(self, state):
        curdepth = state.depth
        if state.success:
            yield state.tid
        for char, substate in state.transitions.items():
            if substate.depth <= curdepth:
                continue
            yield from self._walk_state(substate)

    def push(self, symbol):
        # Just for development
        if len(symbol) > 1:
            result = []
            for c in symbol:
                result.extend(self.push(c))
            return result

        zero_state = self._zero_state
        outbuf = []

        # Process the transition induced by symbol
        state = self.state
        while state is not zero_state:
            if symbol in state.transitions:
                break
            state = state.longest_strict_suffix

        self.state = state.transitions.get(symbol, zero_state)

        self.buf.append(symbol)
        self.idx += 1
        root_idx = self.idx - self.state.depth

        # Update the chain with matches
        for match_state, lidx in self._enumerate_matches(self.state):
            # Find matches that would be interrupted by this match and remove them
            prev_match = None
            for i, match in enumerate(self.chain):
                prev_r = self.base_idx if prev_match is None else prev_match.r
                if prev_r <= lidx <= match.l:
                    for _ in range(len(self.chain) - i):
                        self.chain.pop()
                    break

                prev_match = match

            # If this match is valid, then add it to the chain
            rightmost_r = self.chain[-1].r if self.chain else self.base_idx
            if rightmost_r <= lidx:
                if self.chain:
                    scp, last_r = self.chain[-1].scp.fork(), self.chain[-1].r
                else:
                    # here, we can push the output out immediately
                    # because we know for sure a split is happening here
                    # print(f"{self.buf[self.base_idx:lidx]=}")
                    for c in self.buf[
                        self.base_idx - self.buf_idx : lidx - self.buf_idx
                    ]:
                        outbuf.extend(self.base_scp.push(c))
                    outbuf.extend(self.base_scp.split())

                    self.base_idx = lidx
                    scp, last_r = self.base_scp.fork(), self.base_idx

                gap_outbuf = []
                for char in self.buf[last_r - self.buf_idx : lidx - self.buf_idx]:
                    # print(f"inner push {last_r=} {lidx=} {char=}")
                    gap_outbuf.extend(scp.push(char))

                gap_outbuf.extend(scp.split())

                self.chain.append(
                    self.Match(
                        l=lidx,
                        r=self.idx,
                        tid=match_state.tid,
                        outbuf=gap_outbuf,
                        scp=scp,
                        state=match_state,
                    )
                )
                # this token spans from lidx to idx so no shorter token matches
                break

        # Pull any fully determined added tokens off the chain
        while self.chain and (
            self.chain[0].l < root_idx
            or self.chain[0].l <= root_idx
            and self.state.terminal
        ):
            match = self.chain.popleft()
            outbuf.extend(match.outbuf)
            outbuf.append(match.tid)
            self.base_scp = match.scp
            self.base_idx = match.r

            # TODO: revisit this logic
            # There are two cases to consider: (1) the next token is fixed
            # and (2) only the left edge split is fixed
            if self.chain and self.chain[0].l <= root_idx:
                # dump the pending outbuf of the next chain, since we
                # know its leading split will happen
                match = self.chain[0]
                # print(f"pull_next {gap_outbuf=} {tid=}")
                outbuf.extend(match.outbuf)
                match.outbuf.clear()
                self.base_scp = match.scp
                self.base_idx = match.l

        # Now pull the automaton forward
        while self.base_idx > root_idx:
            self.state = self.state.longest_strict_suffix
            root_idx = self.idx - self.state.depth

        # Fast forward the base_scp to the (new) root of the trie
        if self.base_idx < root_idx:
            for char in self.buf[
                self.base_idx - self.buf_idx : root_idx - self.buf_idx
            ]:
                tmp = self.base_scp.push(char)
                outbuf.extend(tmp)
            self.base_idx = root_idx

        # Pull old chars off the buf as well
        while self.buf_idx < self.base_idx:
            self.buf_idx += 1
            self.buf.popleft()

        return outbuf

    def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
        pointer = tree = {}
        last_pointer = None
        scp, r = self.base_scp, self.base_idx

        # Advance the state according to the chain
        for chain_match in self.chain:
            scp, r = chain_match.scp, chain_match.r

            # Need to "fast forward" the state to reflect the rest of the
            # prefix + suffix. This is still an overapproximation, since
            # some of these may violate the lefmost-longest match semantics
            replay_state = chain_match.state
            # for b in it.chain(self.buf[r - self.buf_idx :], suffix):
            #     new_state = replay_state.transitions[b]
            #     # We know we'll never take a shortcut transition
            #     assert new_state.depth > replay_state.depth
            #     replay_state = new_state

            for b in it.chain(self.buf[r - self.buf_idx :], suffix):
                # If there's no outbound transition, then there's no matches here
                if (new_state := replay_state.transitions.get(b)) is None:
                    break

                # If we take a shortcut, then there's no matches here
                if new_state.depth <= replay_state.depth:
                    break

                replay_state = new_state
            else:
                pointer[None] = torch.tensor(
                    [
                        tid
                        for tid in self._walk_state(replay_state)
                        if tid != chain_match.tid
                    ],
                    device=self.tcs.device,
                )

            for tid in it.chain(chain_match.outbuf, (chain_match.tid,)):
                last_pointer, pointer = pointer, pointer.setdefault(tid, {})

        # Handle future partial matches
        state = self.state
        forked = False
        while True:
            outbuf = []
            root_idx = self.idx - state.depth

            # Advance until we are past the end of the chain match
            if root_idx < r:
                state = state.longest_strict_suffix
                continue

            # Fast forward scp to root_idx
            for c in self.buf[r - self.buf_idx : root_idx - self.buf_idx]:
                # Fork lazily only if we need to push
                if not forked:
                    scp = scp.fork()  # should already be split
                    forked = True
                outbuf.extend(scp.push(c))
            r = root_idx

            # Advance the pointer to reflect the outbuf
            for tid in outbuf:
                last_pointer, pointer = pointer, pointer.setdefault(tid, {})
            split_pointer = pointer

            # Add the special tokens that potentially match from here
            if state is self._zero_state:
                break

            split_scp = scp.fork()
            split_outbuf = split_scp.split()
            for tid in split_outbuf:
                split_pointer = split_pointer.setdefault(tid, {})

            assert None not in split_pointer, f"{split_pointer}"
            # Need to "fast forward" the state to reflect the rest of the
            # suffix. This is still an overapproximation, since
            # some of these may violate the lefmost-longest match semantics
            suffix_state = state
            for b in suffix:
                # If there's no outbound transition, then there's no matches here
                if (new_state := suffix_state.transitions.get(b)) is None:
                    break

                # If we take a shortcut, then there's no matches here
                if new_state.depth <= suffix_state.depth:
                    break

                suffix_state = new_state
            else:
                split_pointer[None] = torch.tensor(
                    list(self._walk_state(suffix_state)), device=self.tcs.device
                )

            state = state.longest_strict_suffix

        assert r == self.idx
        pointer.update(
            scp.eval_tree(
                suffix=suffix, inclusive=inclusive, filter_tensors=filter_tensors
            )
        )

        # Move the terminating branch into the tensor if inclusive is False
        if last_pointer and not pointer:
            # check carefully: tid has not been overwritten
            last_pointer.pop(tid)
            last_pointer[None] = torch.tensor(
                (
                    ([tid] + tids.tolist())
                    if (tids := last_pointer.get(None)) is not None
                    else [tid]
                ),
                device=self.tcs.device,
            )

        return tree

    def __str__(self):
        return f"StreamingAddedTokens({self.chain!r})"
