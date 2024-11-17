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
def generate_diff_report(left_text: str, right_text: str) -> str:
    left_tokens = tokenizer.tokenize(left_text)
    right_tokens = tokenizer.tokenize(right_text)
    additions, subtractions, movements = get_text_deltas(left_tokens, right_tokens)
    
    report = []
    report.append(f'ADDED WORD COUNT (total moved blocks + total added words)\nTotal\t\t{len(additions) + len(movements)}')
    report.append('\n----------------------------------------------------------------------\n')
    report.append(f'ADDED WORDS\nTotal\t\t{len(additions)}')
    report.append(json.dumps(to_text(additions)))
    report.append(f'REMOVED WORDS\nTotal\t\t{len(subtractions)}')
    report.append(json.dumps(to_text(subtractions)))
    report.append(f'MOVED BLOCKS\nTotal\t\t{len(movements)}')
    report.append(json.dumps(to_text([movement[0] for movement in movements]), indent=2))
    
    return '\n'.join(report)
