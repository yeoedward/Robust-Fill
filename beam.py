from heapq import heappush, heappushpop
from typing import List, Tuple
import torch
import torch.nn.functional as F
from tokens import EOS, Tokenizer
from robust_fill import RobustFill


def add_cand(heap: List, width: int, cand: Tuple) -> None:
    """Add candidate to the heap, keeping only the top `width` candidates."""
    if len(heap) < width:
        heappush(heap, cand)
    else:
        heappushpop(heap, cand)


def beam_search(
        model: RobustFill,
        tokenizer: Tokenizer,
        width: int,
        max_program_length: int,
        strings: List[Tuple[List[int], List[int]]]) -> List[Tuple]:
    """
    Beam search for the best program given a list of input-output pairs.

    This is a breadth-first search where the size of the frontier is
    constrainted to the beam width.

    :param model: RobustFill model.
    :param tokenizer: Tokenizes the input and output strings.
    :param width: Beam width to use for decoding.
    :param max_program_length: Limit on length of the programs to search.
    :param strings: List of input-output pairs.
    :returns: Top `width` programs and their scores.
    """
    num_examples = len(strings)
    str_tokens = [
        (tokenizer.tokenize_string(input_), tokenizer.tokenize_string(output))
        for input_, output in strings
    ]
    hidden, all_hidden = model.encode([str_tokens])
    candidates = [(0, [], hidden)]

    for _ in range(max_program_length):
        new_cands = []
        for cand in candidates:
            logit, decoder_input, hidden = cand
            with torch.no_grad():
                input_ = None
                if len(decoder_input) > 0:
                    prev = decoder_input[-1:]
                    if prev[0] == tokenizer.op_token_table[EOS]:
                        # Program already complete, don't need to decode
                        # the next token.
                        add_cand(new_cands, width, cand)
                        continue

                    # Use the previous output as the input to the decoder.
                    input_ = F.one_hot(
                        torch.LongTensor(prev),
                        num_classes=len(tokenizer.op_token_table))
                    # We have to repeat the input for each example
                    # due to the max-pooling in the decoder.
                    input_ = input_.repeat(num_examples, 1)

                logits, new_hidden = model.program_decoder.decode(
                    input_=input_,
                    hidden=hidden,
                    output_all_hidden=all_hidden,
                    num_examples=num_examples,
                )
                topk = torch.topk(logits, k=width, dim=1)

                # Update the frontier.
                for i in range(topk.indices.size()[1]):
                    new_prog = decoder_input + [topk.indices[0, i].item()]
                    try:
                        tokenizer.parse_program(new_prog)
                    except (IndexError, TypeError, ValueError):
                        # Discard invalid programs.
                        continue
                    nc = (
                        logit + topk.values[0, i].item(),
                        new_prog,
                        new_hidden,
                    )
                    add_cand(new_cands, width, nc)

        candidates = new_cands

    return candidates
