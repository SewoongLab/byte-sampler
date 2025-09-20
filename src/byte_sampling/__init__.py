import uuid
from typing import Union

import torch

from .byte_conditioning import ByteConditioning
from .utils import sample_from_logits, sample_from_prob_tree


class EnsembleBytewiseSamplerFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return EnsembleBytewiseSampler(batch_size, *self.args, **self.kwargs)


class EnsembleBytewiseSampler:
    def __init__(self, batch_size, bcs: list[ByteConditioning], mode="mix", **kwargs):
        self.batch_size = batch_size
        self.bcs = bcs
        self.bss = [bc.get_bytewise_sampler(self.batch_size) for bc in bcs]
        self.mode = mode
        self.kwargs = kwargs

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)

    def get_dists(self, **kwargs):
        logits = torch.stack([bs.get_dists(**kwargs) for bs in self.bss], 0).moveaxis(
            1, 0
        )
        logprobs = torch.log_softmax(logits, -1)
        if self.mode == "mix":
            return torch.log_softmax(torch.logsumexp(logprobs, 1), 1)
        elif self.mode == "product":
            power = self.kwargs.get("power", 1 / len(self.bss))
            return torch.log_softmax(logprobs.sum(1) * power, 1)


class BytewisePromptTemplateFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(batch_size, *self.args, **self.kwargs)


class BytewisePromptTemplate:
    def __init__(self, batch_size, bc, prefix, suffix, **kwargs):
        self.batch_size = batch_size
        self.bc = bc
        self.bs = bc.get_bytewise_sampler(batch_size)
        self.rcm = self.bs.rcm
        self.kwargs = kwargs
        self.prompt_added = False
        self.template_prefix, self.template_suffix = prefix, suffix

    def add_context(self, prompts: list[Union[str, bytes]]):
        if not self.prompt_added and self.template_prefix is not None:
            batch = [self.template_prefix] * self.batch_size
            if isinstance(self.template_prefix, (str, bytes)):
                self.bs.add_context(batch)
            else:
                self.bs.add_special_context(batch)

        self.bs.add_context(prompts)

        if not self.prompt_added and self.template_suffix is not None:
            batch = [self.template_suffix] * self.batch_size
            if isinstance(self.template_suffix, (str, bytes)):
                self.bs.add_context(batch)
            else:
                self.bs.add_special_context(batch)

            self.prompt_added = True

    def get_dists(self, **kwargs):
        return self.bs.get_dists(**kwargs)


class BytewiseInstructFactory:
    def __init__(self, bc, extra_suffix="", *args, **kwargs):
        self.bc = bc
        self.args = args
        self.kwargs = kwargs
        self.prefix, self.suffix = self.extract_chat_template(extra_suffix)

    def extract_chat_template(self, extra_suffix):
        sentinel = str(uuid.uuid4())
        template = self.bc.tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix, suffix = template.split(sentinel)
        return self.bc.tokenizer.encode(
            prefix, add_special_tokens=False
        ), self.bc.tokenizer.encode(suffix + extra_suffix, add_special_tokens=False)

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(
            batch_size, self.bc, self.prefix, self.suffix, *self.args, **self.kwargs
        )


class BytewiseQAFactory:
    """Options for QA format:
    qa: Question: {question}\nAnswer: {answer}
    qnan: Question:\n{question}\nAnswer:\n{answer}
    qna: Question:\n{question}\nAnswer: {answer}
    q: Question: {question} (if answer=None, else equivalent to qa)
    """

    def __init__(self, bc, mode="qa", *args, **kwargs):
        self.bc = bc
        self.args = args
        self.kwargs = kwargs

        if mode == "qa":
            self.prefix, self.suffix = "Question: ", "\nAnswer: "
        elif mode == "qnan":
            self.prefix, self.suffix = "Question:\n", "\nAnswer:\n"
        elif mode == "qna":
            self.prefix, self.suffix = "Question:\n", "\nAnswer: "
        else:
            raise NotImplementedError(f"Unknown mode {mode!r}")

    def get_bytewise_sampler(self, batch_size):
        return BytewisePromptTemplate(
            batch_size, self.bc, self.prefix, self.suffix, *self.args, **self.kwargs
        )


class BytewiseProxyTuningFactory:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_bytewise_sampler(self, batch_size):
        return BytewiseProxyTuning(batch_size, *self.args, **self.kwargs)


class BytewiseProxyTuning:
    def __init__(
        self, batch_size, bc_base, bc_expert, bc_antiexpert, alpha=1, **kwargs
    ):
        self.batch_size = batch_size
        self.bc_base = bc_base
        self.bc_expert = bc_expert
        self.bc_antiexpert = bc_antiexpert
        self.bs_base = bc_base.get_bytewise_sampler(batch_size=batch_size)
        self.bs_expert = bc_expert.get_bytewise_sampler(batch_size=batch_size)
        self.bs_antiexpert = bc_antiexpert.get_bytewise_sampler(batch_size=batch_size)
        self.bss = [self.bs_base, self.bs_expert, self.bs_antiexpert]
        self.kwargs = kwargs
        self.alpha = alpha

    @staticmethod
    def extract_chat_template(tokenizer):
        sentinel = str(uuid.uuid4())
        template = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix, suffix = template.split(sentinel)
        return tokenizer.encode(prefix), tokenizer.encode(suffix)

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)

    def get_dists(self, **kwargs):
        logits = torch.stack([bs.get_dists(**kwargs) for bs in self.bss], 0).moveaxis(
            1, 0
        )
        logprobs = torch.log_softmax(logits, -1)
        # Do the proxy tuning!
        # print(self.alpha)
        return torch.log_softmax(
            logprobs[:, 0, :] + (logprobs[:, 1, :] - logprobs[:, 2, :]) * self.alpha, 1
        )


@torch.inference_mode()
def generate_batched(
    sampler_factory,
    prompts: list[str],
    max_new_bytes: int = 100,
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    display: bool = False,
    stop_strings: tuple[str] = (),
    include_stop_str_in_output: bool = False,
    allow_special: bool = True,
    logprob_transforms=None,
):
    assert not isinstance(
        stop_strings, str
    ), "stop_strings should be a sequence of strings"
    stop_strings = tuple(sorted(stop_strings, key=len, reverse=True))
    assert not isinstance(prompts, str)
    assert seed is None or generator is None, "can pass only one of seed/generator"

    bsize = len(prompts)
    assert not (display and bsize > 1)

    try:
        bs = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        bs = sampler_factory

    bs.add_context([prompt.encode() for prompt in prompts])

    outputs = [[] for _ in range(bsize)]
    decode_bufs = [b"" for _ in range(bsize)]
    stop_found = [False for _ in range(bsize)]

    if display:
        print(prompts[0], end="", flush=True)

    for _ in range(max_new_bytes):
        dists = bs.get_dists(logprob_transforms=logprob_transforms)
        if not allow_special:
            dists[:, 256:] = -torch.inf

        # init the generator late so we know which device to put it on
        if generator is None and seed is not None:
            generator = torch.Generator(device=dists.device).manual_seed(seed)

        new_bytes = sample_from_logits(
            dists,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).tolist()

        for i, new_byte in enumerate(new_bytes):
            if new_byte >= 256:
                stop_found[i] = True

        new_bytes = [
            bytes([b]) if not sf else bytes() for b, sf in zip(new_bytes, stop_found)
        ]

        bs.add_context(new_bytes)

        for i, new_byte in enumerate(new_bytes):
            if stop_found[i]:
                continue
            try:
                decode_bufs[i] += new_byte
                char = decode_bufs[i].decode()
                outputs[i].append(char)
                if display:
                    print(char, end="", flush=True)
                decode_bufs[i] = b""
            except UnicodeDecodeError:
                pass

        if stop_strings:
            for i, output in enumerate(outputs):
                if stop_found[i]:
                    continue

                suffix = "".join(output[-max(map(len, stop_strings)) :])
                if suffix.endswith(stop_strings):
                    if not include_stop_str_in_output:
                        for stop in stop_strings:
                            if suffix.endswith(stop):
                                outputs[i] = output[: -len(stop)]
                                break

                    stop_found[i] = True

        if all(stop_found):
            break

    # print(decode_bufs)
    return ["".join(output) for output in outputs]


@torch.inference_mode()
def generate_pbp_prepare(
    sampler_factory,
    prompts: list[str],
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
):
    assert not isinstance(prompts, str)
    bsize = len(prompts)

    try:
        sampler = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        sampler = sampler_factory

    sampler.add_context([prompt.encode() for prompt in prompts])

    eval_trees, prob_trees = sampler.tree_inference(
        inclusive=False, filter_tensors=True, do_gc=False
    )

    sampled_seqs = [
        trunk
        + sample_from_prob_tree(
            et,
            pt,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        for trunk, et, pt in zip(sampler.trunks, eval_trees, prob_trees)
    ]

    past_key_values = sampler.rcm.export_cache(sampled_seqs, inplace=False)

    maxlen = max(map(len, sampled_seqs))
    input_ids = torch.tensor(
        [[sampler.bc.pad] * (maxlen - len(seq)) + seq for seq in sampled_seqs],
        dtype=torch.long,
        device=sampler.bc.device,
    )

    attention_mask = torch.tensor(
        [[0] * (maxlen - len(seq)) + [1] * len(seq) for seq in sampled_seqs],
        dtype=torch.bool,
        device=sampler.bc.device,
    )

    return input_ids, attention_mask, past_key_values


@torch.inference_mode()
def generate_pbp_batched(
    sampler_factory,
    prompts: list[str],
    do_sample: bool = True,
    temperature: float = 1,
    top_k: float | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    generate_kwargs: dict | None = None,
    max_new_tokens: int = 20,
):
    bsize = len(prompts)

    try:
        sampler = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        sampler = sampler_factory

    assert do_sample or top_k is None
    assert do_sample or top_p is None

    input_ids, attention_mask, past_key_values = generate_pbp_prepare(
        sampler_factory=sampler,
        prompts=prompts,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )

    full_generate_kwargs = dict(
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    if generate_kwargs is not None:
        full_generate_kwargs.update(generate_kwargs)

    return sampler.bc.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        **full_generate_kwargs,
    )
