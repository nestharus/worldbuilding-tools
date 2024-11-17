import json

from text_comparator.get_text_diff import get_text_deltas
from tokenizer.context_aware_tokenizer import ContextAwareTokenizer
from tokenizer.deberta_tokenizer import DebertaTokenizer
from tokenizer.spacy_tokenizer import spacy_tokenizer


def text_tokens(tokenizer):
    return lambda tokens: [
        tokenizer.deberta_tokenizer.tokenizer.convert_ids_to_tokens(token[2]) if isinstance(token[2], int) else '[' + ','.join(tokenizer.deberta_tokenizer.tokenizer.convert_ids_to_tokens(token[2])) + ']'
        for token in tokens
    ]


tokenizer = ContextAwareTokenizer(DebertaTokenizer(), spacy_tokenizer())
to_text = text_tokens(tokenizer)
with open('original.txt', 'r') as file:
    left_text = file.read()
with open('revised.txt', 'r') as file:
    right_text = file.read()

left_tokens = tokenizer.tokenize(left_text)
right_tokens = tokenizer.tokenize(right_text)
additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)

with open('report.txt', 'w') as file:
    file.write(f'ADDED WORD COUNT (total moved blocks + total added words)\nTotal\t\t{len(additions) + len(movements)}\n')
    file.write('\n')
    file.write('----------------------------------------------------------------------\n')
    file.write('\n')
    file.write(f'ADDED WORDS\nTotal\t\t{len(additions)}\n')
    file.write(json.dumps(to_text(additions)) + '\n')
    file.write(f'REMOVED WORDS\nTotal\t\t{len(subtractions)}\n')
    file.write(json.dumps(to_text(subtractions)) + '\n')
    file.write(f'MOVED BLOCKS\nTotal\t\t{len(movements)}\n')
    file.write(json.dumps(to_text([movement[0] for movement in movements]), indent=2) + '\n')
