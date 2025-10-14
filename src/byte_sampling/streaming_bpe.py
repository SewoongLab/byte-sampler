from copy import copy
from dataclasses import dataclass
from typing import Optional, Self

import torch


class StreamingBPE:
    @dataclass(slots=True)
    class Node:
        last_tid: Optional[int]
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
        self.last_heads = [self.tree]

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
                if head.last_tid is None or self.tcs._valid_adj(head.last_tid, newtid):
                    newhead = self.Node(newtid, head, {}, self.tcs.vtrie, [])
                    head.children[newhead.last_tid] = newhead
                    new_heads.append(newhead)
                    heads_created.append(newhead)

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
        new_heads = []
        new_last_heads = []

        def copy_tree(node: self.Node, parent: Optional[self.Node] = None):
            nonlocal new_last_heads
            newnode = self.Node(
                node.last_tid, parent, None, node.trie, copy(node.trie_path)
            )
            if node in self.heads:
                new_heads.append(newnode)
            if node in self.last_heads:
                new_last_heads.append(newnode)
            newnode.children = {
                tid: copy_tree(child, newnode) for tid, child in node.children.items()
            }
            return newnode

        new.tree = copy_tree(self.tree)
        new.heads = new_heads
        new.last_heads = new_last_heads

        return new

    def eval_tree(self, suffix=b"", inclusive=False, filter_tensors=True):
        if suffix:
            # for convenience
            self_copy = self.fork()
            copy_tokens = []
            for b in suffix:
                copy_tokens.extend(self_copy.push(b))
            tree = self_copy.eval_tree(
                inclusive=inclusive, filter_tensors=filter_tensors
            )
            for tid in reversed(copy_tokens):
                tree = {tid: tree}
            return tree

        def convert_tree(node: self.Node):
            converted_node = {}
            if node.trie is not None:
                if filter_tensors:
                    # print(f"tree: {node.last_tid}, {bytes(node.trie_path) + suffix!r}")
                    valid_tokens = self.tcs._valid_r_filtered(
                        node.last_tid, bytes(node.trie_path)
                    )
                else:
                    valid_tokens = self.tcs._valid_r_unfiltered(node.trie_path)

                if len(valid_tokens) > 0:
                    converted_node[None] = valid_tokens

            for tid, child in node.children.items():
                subtree, was_last = convert_tree(child)
                if subtree:
                    converted_node[tid] = subtree
                    if (
                        valid_tokens := converted_node.get(None)
                    ) is not None and was_last:
                        converted_node[None] = valid_tokens[valid_tokens != tid]
                        if len(converted_node[None]) == 0:
                            converted_node.pop(None)

            if node in self.last_heads:
                if not inclusive:
                    return {}, True
                else:
                    converted_node[None] = torch.arange(
                        self.tcs.tokenizer.vocab_size, device=self.tcs.device
                    )
                    # converted_node[None] = slice(len(self.tcs.vocab))

            return converted_node, node in self.last_heads

        converted_tree, _ = convert_tree(self.tree)
        return converted_tree
