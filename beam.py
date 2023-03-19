from heapq import heappush, heappushpop
from typing import List, Tuple
import torch
import torch.nn.functional as F
from tokens import Tokenizer
from robust_fill import RobustFill


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
                    # Use the previous output as the input to the decoder.
                    input_ = F.one_hot(
                        torch.LongTensor(decoder_input[-1:]),
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
                for i in range(width):
                    nc = (
                        logit + topk.values[0, i].item(),
                        decoder_input + [topk.indices[0, i].item()],
                        new_hidden,
                    )
                    if len(new_cands) < width:
                        heappush(new_cands, nc)
                    else:
                        heappushpop(new_cands, nc)
        candidates = new_cands

    return candidates
