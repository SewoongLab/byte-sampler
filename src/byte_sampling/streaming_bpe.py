from copy import copy
from dataclasses import dataclass
from typing import Optional, Self

import torch


class StreamingBPE:
    @dataclass(slots=True, eq=False)
    class Node:
        last_tid: Optional[int]
        parent: Optional[Self]
        children: dict[int, Self]  # tid -> child
        trie: Optional[dict]
        trie_path: list

        def __repr__(self):
            return f"N({self.last_tid})"

        __str__ = __repr__

        # Use identity-based equality and hashing for sets
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    def __init__(self, tcs: "ByteConditioning"):
        self.tcs = tcs
        self.reset()
        self.total_time = 0

    def reset(self):
        self.tree = self.Node(None, None, {}, self.tcs.vtrie, [])
        self.heads = {self.tree}
        self.last_heads = {self.tree}

    def gc_node(self, node):
        if node.parent is not None and not node.children and node.trie is None:
            # print(f"removing node {node}")
            node.parent.children.pop(node.last_tid)
            self.gc_node(node.parent)

    def push(self, byte: int):
        assert isinstance(byte, int)
        new_heads = set()
        fixed_tokens = []

        # The core streaming update
        heads_created = set()
        for head in self.heads:
            if byte not in head.trie:
                # this head has "died"
                head.trie = None
                self.gc_node(head)
                continue

            trie = head.trie = head.trie[byte]
            head.trie_path.append(byte)
            new_heads.add(head)
            if (newtid := trie.get(None)) is not None:
                # if head.parent is None or self.tcs._valid_adj(head.last_tid, newtid):
                if head.last_tid is None or self.tcs._valid_adj(head.last_tid, newtid):
                    newhead = self.Node(newtid, head, {}, self.tcs.vtrie, [])
                    head.children[newhead.last_tid] = newhead
                    new_heads.add(newhead)
                    heads_created.add(newhead)

        # Some quick sanity checks
        def trace_path(head):
            pathrev = []
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
        assert (
            len(heads_created) >= 1
        ), f"sequence ending in {bytes([byte])!r} cannot be tokenized"

        self.heads = new_heads
        self.last_heads = heads_created

        # Pull off any root node with unambiguous children
        while len(self.tree.children) == 1:
            if self.tree.trie is not None:
                break

            new_root = next(iter(self.tree.children.values()))
            new_root.parent = None
            self.tree = new_root
            fixed_tokens.append(self.tree.last_tid)

        return fixed_tokens

    def split(self):
        if self.tcs.has_ignore_merges and len(self.last_heads) > 1:
            # If there's multiple paths to the same byte, one must be through the ignored merge
            assert len(self.last_heads) <= 2
            unreachable_heads = [
                head for head in self.last_heads if head.parent.last_tid is None
            ]
            assert len(unreachable_heads) == 1
            self.reset()
            return [unreachable_heads[0].last_tid]

        assert len(self.last_heads) == 1
        pointer = next(iter(self.last_heads))
        path_rev = []
        while pointer.parent is not None:
            path_rev.append(pointer.last_tid)
            pointer = pointer.parent
        self.reset()
        return path_rev[::-1]

    def fork(self):
        new = self.__class__(self.tcs)
        # deepcopy the tree but NOT the trie!
        new_heads = set()
        new_last_heads = set()

        def copy_tree(node: self.Node, parent: Optional[self.Node] = None):
            nonlocal new_last_heads
            newnode = self.Node(
                node.last_tid, parent, None, node.trie, copy(node.trie_path)
            )
            if node in self.heads:
                new_heads.add(newnode)
            if node in self.last_heads:
                new_last_heads.add(newnode)
            newnode.children = {
                tid: copy_tree(child, newnode) for tid, child in node.children.items()
            }
            return newnode

        new.tree = copy_tree(self.tree)
        new.heads = new_heads
        new.last_heads = new_last_heads

        return new

    @classmethod
    def tree_update(cls, tree1, tree2):
        "Merge tree2 into tree1. Mutates tree1."
        merged = tree1
        for tid, subtree in tree2.items():
            if tid in merged:
                if tid is None:
                    # pure torch to avoid a device sync
                    merged[tid] = torch.cat((merged[tid], subtree)).unique()
                else:
                    merged[tid] = cls.tree_update(merged[tid], subtree)
            else:
                merged[tid] = subtree

        return merged

    def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
        if suffix:
            # Suffix may be (at most) a partial character this means that
            # we can't trust StreamingCharPretok to predict whether there
            # is a split at the beginning of the suffix. Thus, we handle
            # both cases here.
            def build_tree_from_suffix(sbpe, split):
                outbuf = []
                if split:
                    outbuf.extend(sbpe.split())
                for b in suffix:
                    outbuf.extend(sbpe.push(b))
                tree = sbpe.eval_tree(
                    inclusive=inclusive, filter_tensors=filter_tensors
                )
                for tid in reversed(outbuf):
                    tree = {tid: tree}
                return tree

            unsplit_tree = build_tree_from_suffix(self.fork(), False)
            split_tree = build_tree_from_suffix(self.fork(), True)
            return self.tree_update(unsplit_tree, split_tree)

        def convert_tree(node: self.Node):
            converted_node = {}

            if node in self.last_heads:
                if not inclusive:
                    return {}
                else:
                    converted_node[None] = self.tcs.token_slicer.all()

            if node.trie is not None:
                prefix = bytes(node.trie_path)
                if filter_tensors:
                    valid_tokens = self.tcs._valid_r_filtered(node.last_tid, prefix)
                else:
                    valid_tokens = self.tcs._valid_r_unfiltered(prefix)

                if valid_tokens.numel():
                    converted_node[None] = valid_tokens

            for tid, child in node.children.items():
                subtree = convert_tree(child)
                if subtree:
                    converted_node[tid] = subtree

            return converted_node

        converted_tree = convert_tree(self.tree)
        return converted_tree
