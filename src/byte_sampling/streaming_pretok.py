import itertools as it
import re
import string
from collections import deque
from copy import copy

import torch

from .streaming_bpe import StreamingBPE


class PretokExceptionHandler:
    def __init__(self, pre_tokenizer):
        self.pre_tokenizer = pre_tokenizer
        self.pre_tokenize = self.pre_tokenizer.pre_tokenize_str

    def pretok_pattern(self, s):
        return tuple([split for _, (_, split) in self.pre_tokenize(s)[:-1]])

    def get_splits(self, s):
        raise NotImplementedError


class ContractionHandler(PretokExceptionHandler):
    CONTRACTIONS = ("re", "ve", "ll")

    def __init__(self, pre_tokenizer):
        super().__init__(pre_tokenizer)

        witnesses = "zq"
        default_patterns = {self.pretok_pattern(f"{w}'{w}") for w in witnesses}
        assert len(default_patterns) == 1

        self.default_pattern = next(iter(default_patterns))
        assert self.default_pattern in (
            (),
            (1,),
            (1, 2),
        ), "Unknown default contraction behavior"

        # As far as we know, the left side doesn't matter
        witnesses = "zqiI"
        self.exceptions = {}

        for suffix in it.chain(
            string.ascii_letters,
            it.chain.from_iterable(
                self.case_variations(suffix) for suffix in self.CONTRACTIONS
            ),
        ):
            patterns = {self.pretok_pattern(f"{w}'{suffix}") for w in witnesses}
            assert len(patterns) == 1
            pattern = next(iter(patterns))
            if pattern != self.default_pattern:
                self.exceptions[suffix] = pattern

        witness = "/!_"
        self.has_punct_exception = None
        check = {
            self.pretok_pattern(f"{w1}'{w2}") for w1 in " " + witness for w2 in witness
        }
        assert len(check) == 1
        check = next(iter(check))
        if check != self.default_pattern:
            self.has_punct_exception = check

    @staticmethod
    def case_variations(s: str):
        """
        Yield all case variations of `s`, toggling only characters with
        a 1-to-1 lower/upper mapping (avoids length-changing Unicode cases
        like ß -> SS). Non-letters are emitted as-is.

        Complexity: O(2^M) outputs, where M is the number of togglable letters.
        """

        def _gen(i: int, prefix: str):
            if i == len(s):
                yield prefix
                return
            ch = s[i]
            lo, up = ch.lower(), ch.upper()
            # Toggle only if the mapping is a true 1-to-1 case pair
            if lo != up and len(lo) == 1 and len(up) == 1:
                # lower branch
                yield from _gen(i + 1, prefix + lo)
                # upper branch
                yield from _gen(i + 1, prefix + up)
            else:
                # leave as-is (digits, punctuation, or length-changing case)
                yield from _gen(i + 1, prefix + ch)

        return _gen(0, "")

    def get_splits(self, s):
        if not self.exceptions or "'" not in s[-2:]:
            return

        def shift_and_trim(idx, pat):
            shifted = (split - (idx + 1) for split in pat)
            return tuple(i for i in shifted if -len(s) < i < 0)

        for i in range(
            1, min(len(s) + 1, max(len(suffix) for suffix in self.exceptions) + 1)
        ):
            if s[-i] != "'":
                continue

            # Compute the contractions that may be completed by the current prefix
            matches = {
                suffix: p
                for suffix, p in self.exceptions.items()
                if len(suffix) > (i - 1) and suffix.startswith(s[len(s) - i + 1 :])
            }
            # print(f"{i=} {matches=}")

            if matches:
                default_shifted = shift_and_trim(i, self.default_pattern)
                result = {default_shifted}

                # Punctuation on the left of the apostrophe can avoid the split
                if self.has_punct_exception is not None:
                    punct_shifted = shift_and_trim(i, self.has_punct_exception)
                    result.add(punct_shifted)
                    if (
                        len(s) > i
                        and i > 1
                        and (
                            s[-i - 1] == " "
                            or not (
                                s[-i - 1].isdigit()
                                or s[-i - 1].isspace()
                                or s[-i - 1].isalpha()
                            )
                        )
                    ):
                        result.add((-i + 1,))

                result.update(shift_and_trim(i, pat) for pat in matches.values())
                return result


class WhitespaceLookaheadHandler(PretokExceptionHandler):
    def __init__(self, pre_tokenizer):
        super().__init__(pre_tokenizer)

        self.has_split = self.pretok_pattern("  a") == (1,)

    def get_splits(self, s):
        if self.has_split and len(s) >= 2 and s[-2:].isspace():
            return {(), (-1,)}


class DigitAlignmentHandler(PretokExceptionHandler):
    def __init__(self, pre_tokenizer):
        super().__init__(pre_tokenizer)
        self.has_right_align = self.pretok_pattern("1234") == (1,)

    def get_splits(self, s):
        if not s or not self.has_right_align:
            return

        i = -1
        while -i <= len(s) and s[i].isdigit():
            i -= 1

        if i > -3:
            return

        return {tuple(range(i + j + 2, 0, 3)) for j in range(3)}


class StreamingCharPretok:
    class TempPretokState:
        __slots__ = ["sbpe", "buf", "outbuf"]

        def __init__(self, sbpe: StreamingBPE, buf="", outbuf=None):
            self.sbpe = sbpe
            self.buf = buf
            self.outbuf = deque() if outbuf is None else outbuf

        def push(self, char):
            self.buf += char
            for b in char.encode():
                self.outbuf.extend(self.sbpe.push(b))
            return self

        def split(self, do_split=True):
            if do_split:
                self.buf = ""
                self.outbuf.extend(self.sbpe.split())
            return self

        def peek(self):
            return self.outbuf

        def pull(self):
            result = self.outbuf
            self.outbuf = deque()
            return result

        def fork(self):
            return self.__class__(self.sbpe.fork(), self.buf, copy(self.outbuf))

        def __repr__(self):
            return f"State({list(self.outbuf)} <- {self.buf!r})"

    def __init__(
        self,
        tcs: "ByteConditioning",
        handlers=None,
        buf=None,
        state=None,
        branches=None,
        active_handler=None,
    ):
        self.tcs = tcs
        self.pretokenize = (
            tcs.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
        )
        self.handlers = (
            [
                handler_factory(tcs.tokenizer.backend_tokenizer.pre_tokenizer)
                for handler_factory in (
                    ContractionHandler,
                    WhitespaceLookaheadHandler,
                    DigitAlignmentHandler,
                )
            ]
            if handlers is None
            else handlers
        )
        self.reset(
            buf=buf,
            state=state,
            branches=branches,
            active_handler=active_handler,
        )

    def __repr__(self):
        return f"SCP({self.buf!r}, {self.branches!r})"

    def reset(self, buf=None, state=None, branches=None, active_handler=None):
        self.buf = "" if buf is None else buf
        self.state = (
            self.TempPretokState(self.tcs.get_streaming_bpe())
            if state is None
            else state
        )
        self.branches = {} if branches is None else branches
        self.active_handler = active_handler

    def _pull(self):
        if not self.branches:
            return self.state.pull()

        outbuf = deque()
        while True:
            # TODO: a dequeue is probably a better data structure for this
            next_toks = {
                next(iter(state.outbuf), None) for state in self.branches.values()
            }
            # print(f"{next_toks=} {[next(iter(state.outbuf), None) for state in self.branches.values()]}")
            if len(next_toks) == 1 and (next_tok := next(iter(next_toks))) is not None:
                outbuf.append(next_tok)
                for state in self.branches.values():
                    state.outbuf.popleft()
            else:
                break

        return outbuf

    def _pretok_splits(self, string):
        pretokens = self.pretokenize(string)
        pretok_splits = tuple(
            [split - len(string) + 1 for _, (_, split) in pretokens[:-1]]
        )
        # print(f"{self.state.buf + char=} {pretokens=} {pretok_splits=}")

        for (_, (_, aend)), (_, (bstart, _)) in it.pairwise(pretokens):
            assert aend == bstart, f"got gap in pretokens: {pretokens}"
        return pretok_splits

    def push(self, char: str):
        if self.tcs.btok.normalizer is not None:
            char = self.tcs.btok.normalizer.normalize_str(char)

        # Just for development
        if len(char) > 1:
            result = deque()
            for c in char:
                result.extend(self.push(c))
            return result

        pretok_splits = self._pretok_splits(self.state.buf + char)
        old_active_handler = self.active_handler
        self.active_handler, detected_splits = None, None

        for i, handler in enumerate(self.handlers):
            if splits := handler.get_splits(self.state.buf + char):
                assert self.active_handler is None
                self.active_handler, detected_splits = i, splits

        # Now, for the four top-level cases
        # print(f"{old_active_handler=}, {self.active_handler=}")
        match old_active_handler, self.active_handler:
            case None, None:
                # Normal case, no ambiguity
                first_split = next(iter(pretok_splits), None)
                assert (
                    first_split is None or first_split >= 0
                ), f"detected unhandled unstable split: {pretok_splits}"
                self.state.split(first_split == 0).push(char)

            case None, _:
                # Rising edge
                assert not self.branches
                # Note, we haven't pushed the new character onto self.sbpe yet
                for case in detected_splits:
                    assert (
                        not case or case[0] >= -1
                    ), f"detected unhandled unstable split: {pretok_splits}"
                    # print(f"{case=}")
                    self.branches[case] = (
                        self.state.fork().split(case and case[-1] == -1).push(char)
                    )

                self.state.buf += char

                # print(self.branches)

            case _, None:
                # print("falling edge", pretok_splits, self.branches)
                key = pretok_splits
                if has_split := key and key[-1] == 0:
                    key = key[:-1]
                # Here, we detect which branch matches what we saw and then select it
                # print(self.state)
                assert key in self.branches, f"{self.branches} missing {key}"
                self.state = self.branches[key].split(has_split).push(char)
                self.branches = {}

            case prv, nxt:
                first_unstable = min(
                    next(iter(case), float("inf")) for case in detected_splits
                )
                prefix = tuple(p - 1 for p in pretok_splits if p - 1 < first_unstable)
                new_branches = {}

                for case in detected_splits:
                    case = (*prefix, *case)
                    # print(case)
                    key = tuple(c + 1 for c in case)
                    # print(f"{key=}")
                    new_branches[case] = (
                        self.branches[tuple(c for c in key if c < 0)]
                        .fork()
                        .split(0 in key)
                        .push(char)
                    )

                self.branches = new_branches
                self.state.buf += char

        return self._pull()

    def split(self):
        if not self.branches:
            state = self.state
        else:
            pretok_splits = self._pretok_splits(self.state.buf)
            key = tuple(p - 1 for p in pretok_splits)
            state = self.branches[key]

        outbuf = state.split().pull()
        self.reset()
        return outbuf

    def fork(self):
        return self.__class__(
            tcs=self.tcs,
            handlers=self.handlers,  # do not need to re-init the handlers!
            buf=self.buf,
            state=self.state.fork(),
            branches={pat: state.fork() for pat, state in self.branches.items()},
            active_handler=self.active_handler,
        )

    def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
        def get_tree(state):
            return state.sbpe.eval_tree(
                suffix=suffix, inclusive=inclusive, filter_tensors=filter_tensors
            )

        if not self.branches:
            return get_tree(self.state)

        def merge_trees(n1, n2):
            merged = n1
            for tid, subtree in n2.items():
                if tid in merged:
                    if tid is None:
                        # pure torch to avoid a device sync
                        merged[tid] = torch.cat((merged[tid], subtree)).unique()
                    else:
                        merged[tid] = merge_trees(merged[tid], subtree)
                else:
                    merged[tid] = subtree

            return merged

        result = None
        for branch_state in self.branches.values():
            subtree = get_tree(branch_state)
            for tid in reversed(branch_state.outbuf):
                subtree = {tid: subtree}
            # print(subtree)
            if result is None:
                result = subtree
            else:
                result = merge_trees(result, subtree)

        return result
