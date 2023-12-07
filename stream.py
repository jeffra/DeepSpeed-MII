"""
This example is mostly borrowed from vllm which was originally borrowed from huggingface

https://github.com/vllm-project/vllm/blob/42c02f5892e984d308614f074f423a311aba8993/vllm/engine/llm_engine.py#L660-L678
https://github.com/vllm-project/vllm/blob/42c02f5892e984d308614f074f423a311aba8993/vllm/transformers_utils/tokenizer.py#L119-L182
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class Sequence:
    """A sequence of tokens that is being decoded incrementally."""
    prefix_offset: int = 0
    read_offset: int = 0
    output_text: str = ""
    token_ids: List[int] = None
    tokens: List[str] = None


def _convert_tokens_to_string_with_added_encoders(
    tokenizer,
    output_tokens: List[str],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    if spaces_between_special_tokens:
        return " ".join(sub_texts)
    else:
        return "".join(sub_texts)
    

# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer,
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int = 0,
    read_offset: int = 0,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    if prev_tokens is None:
        new_tokens = tokenizer.convert_ids_to_tokens(
            all_input_ids, skip_special_tokens=skip_special_tokens)
        output_tokens = new_tokens
        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_offset = max(len(output_tokens) - 6, 0)
        # If the first new token is a special token, we can't skip 1 extra token
        if skip_special_tokens and new_token_id in tokenizer.all_special_ids:
            read_offset = max(len(output_tokens), 0)
        else:
            read_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens)
        output_tokens = prev_tokens + new_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text):]
        return new_tokens, new_text, read_offset, len(output_tokens)
    else:
        return new_tokens, "", prefix_offset, read_offset


def decode_sequence(tokenizer, seq: Sequence) -> None:
    """Decodes the new token for a sequence."""
    (new_tokens, new_output_text, prefix_offset,
        read_offset) = detokenize_incrementally(
            tokenizer,
            all_input_ids=seq.token_ids,
            prev_tokens=seq.tokens,
            prefix_offset=seq.prefix_offset,
            read_offset=seq.read_offset,
            skip_special_tokens=True, #prms.skip_special_tokens,
            spaces_between_special_tokens=True, #prms.spaces_between_special_tokens,
        )
    if seq.tokens is None:
        seq.tokens = new_tokens
    else:
        seq.tokens.extend(new_tokens)
    seq.prefix_offset = prefix_offset
    seq.read_offset = read_offset
    seq.output_text += new_output_text


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    prompt = "DeepSpeed runs on PyTorch"
    encoded = tok.encode(prompt)
    print(encoded)

    seq = Sequence()
    seq.token_ids = []

    for token_id in encoded:
        seq.token_ids.append(token_id)
        decode_sequence(tok, seq)
        print(f"{seq.output_text=}, {seq.tokens=}")

