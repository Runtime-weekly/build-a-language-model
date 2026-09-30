"""Character bigram counts: a transparent baseline, not a transformer."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

CORPUS = ['the cat sat.', 'the dog sat.', 'the cat ran.']


def vocabulary(documents):
    """Sorted unique characters; IDs are zero-based positions in this list."""
    train(documents)  # Reuse corpus validation; no data or model mutation.
    return sorted(set(''.join(documents)))


def validate_vocabulary(tokens):
    if not isinstance(tokens, (list, tuple)) or any(not isinstance(t, str) or len(t) != 1 for t in tokens):
        raise ValueError('Vocabulary must be a list of individual characters')
    if len(tokens) != len(set(tokens)):
        raise ValueError('Vocabulary characters must be unique')


def encode(text, tokens):
    validate_vocabulary(tokens)
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    ids = {character: index for index, character in enumerate(tokens)}
    if any(character not in ids for character in text):
        raise ValueError('Text contains a character outside the vocabulary')
    return [ids[character] for character in text]


def decode(ids, tokens):
    validate_vocabulary(tokens)
    if not isinstance(ids, (list, tuple)) or any(type(i) is not int or not 0 <= i < len(tokens) for i in ids):
        raise ValueError('IDs must be integer vocabulary indices')
    return ''.join(tokens[i] for i in ids)


def shifted(document):
    """Each input character predicts the immediately following character."""
    if not isinstance(document, str):
        raise ValueError('Document must be text')
    return {'inputs': list(document[:-1]), 'targets': list(document[1:])}


def train(documents):
    if not isinstance(documents, list) or not documents or any(not isinstance(d, str) for d in documents):
        raise ValueError('Corpus must be a nonempty list of text documents')
    counts = defaultdict(Counter)
    for document in documents:
        pairs = shifted(document)
        for current, following in zip(pairs['inputs'], pairs['targets']):
            counts[current][following] += 1
    # Never join documents: doing so would invent end-to-start training pairs.
    return {context: dict(sorted(row.items())) for context, row in sorted(counts.items())}


def probabilities(counts, context):
    if not isinstance(context, str) or len(context) != 1:
        raise ValueError('Context must be exactly one character')
    row = counts.get(context, {})
    total = sum(row.values())
    if total == 0:
        return {}  # No smoothing or invented fallback distribution.
    return {token: count/total for token, count in sorted(row.items())}


def generate(counts, prompt='t', max_new_characters=40, seed=7):
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('Prompt must contain at least one character')
    if type(max_new_characters) is not int or max_new_characters < 0:
        raise ValueError('Generation length must be a nonnegative integer')
    if type(seed) is not int:
        raise ValueError('Seed must be an integer')
    rng = random.Random(seed)
    text, steps = prompt, []
    stop_reason = 'length_limit'
    for index in range(max_new_characters):
        context = text[-1]
        if context == '.':
            stop_reason = 'period'
            break
        row = counts.get(context, {})
        total = sum(row.values())
        if not total:
            stop_reason = 'unseen_context'
            break
        # Integer-ticket sampling avoids dependence on rounded probabilities.
        ticket = rng.randrange(total)
        cumulative = 0
        for character, count in sorted(row.items()):
            cumulative += count
            if ticket < cumulative:
                chosen = character
                break
        steps.append({'step': index+1, 'prefix_before': text, 'context': context, 'ticket': ticket,
                      'total_count': total, 'probabilities': probabilities(counts, context),
                      'next_character': chosen, 'prefix_after': text+chosen})
        text += chosen
        if chosen == '.':
            stop_reason = 'period'
            break
    return {'prompt': prompt, 'seed': seed, 'max_new_characters': max_new_characters,
            'text': text, 'generated_characters': len(steps), 'stop_reason': stop_reason,
            'steps': steps}


def heldout_score(training_documents, heldout_documents):
    """Unsmoothed negative log likelihood in nats; never fit heldout pairs."""
    if not isinstance(heldout_documents, list) or not heldout_documents or any(not isinstance(d, str) for d in heldout_documents):
        raise ValueError('Heldout documents must be a nonempty list of texts')
    if set(training_documents) & set(heldout_documents):
        raise ValueError('Training and heldout documents overlap')
    counts = train(training_documents)
    nll, pairs, unseen = 0.0, 0, []
    for document_index, document in enumerate(heldout_documents):
        for index, (context, target) in enumerate(zip(document, document[1:])):
            pairs += 1
            probability = probabilities(counts, context).get(target, 0)
            if probability:
                nll -= math.log(probability)
            else:
                unseen.append({'document_index': document_index, 'position': index,
                               'context': context, 'target': target})
    return {'pair_count': pairs, 'zero_probability_pairs': unseen,
            'mean_nll_nats': nll/pairs if pairs and not unseen else None,
            'status': 'infinite_unsmoothed_loss' if unseen else ('finite' if pairs else 'no_pairs')}


def demo():
    counts = train(CORPUS)
    return {'label': 'Character-count bigram baseline; not a transformer or trained neural network',
            'corpus': CORPUS, 'vocabulary': sorted(set(''.join(CORPUS))),
            'document_boundaries': 'No pairs cross documents; no BOS/EOS tokens added',
            'shifted_examples': [shifted(document) for document in CORPUS],
            'counted_pairs': sum(sum(row.values()) for row in counts.values()),
            'counts': counts, 'probabilities': {context: probabilities(counts, context) for context in counts},
            'inspected_context': {'character': 't', 'counts': counts['t'], 'probabilities': probabilities(counts, 't')},
            'row_a': {'character': 'a', 'counts': counts['a'], 'probabilities': probabilities(counts, 'a')},
            'row_a_generation': generate(counts, prompt='ca', max_new_characters=6, seed=0),
            'encoding_example': {'vocabulary': vocabulary(CORPUS), 'text': 'cat',
                                 'ids': encode('cat', vocabulary(CORPUS)),
                                 'decoded': decode(encode('cat', vocabulary(CORPUS)), vocabulary(CORPUS))},
            'generation': generate(counts),
            'unseen_context_demo': generate(counts, prompt='?'),
            'heldout_example': {'training_documents': CORPUS[:2], 'heldout_documents': CORPUS[2:],
                                'result': heldout_score(CORPUS[:2], CORPUS[2:])},
            'limitations': ['Only the previous character is used; longer prompts do not add context.',
                            'No smoothing: missing transitions have probability zero.',
                            'Generation stops explicitly at a period, at the length cap, or at a context without outgoing counts; period stop is a teaching rule, not a learned EOS token.',
                            'Generation model uses all three documents; heldout scoring separately fits only the first two.',
                            'Tiny authored corpus and one heldout document are educational, not a generalization benchmark.',
                            'Seeded sampling is repeatable within the tested Python implementation; trace includes exact random tickets.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('demo', 'inspect', 'generate'), nargs='?', default='demo')
    parser.add_argument('--corpus', type=Path, help='Optional UTF-8 JSON list of documents')
    parser.add_argument('--context', default='t')
    parser.add_argument('--prompt', default='t')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--length', type=int, default=40)
    args = parser.parse_args()
    if args.command == 'demo':
        if args.corpus:
            parser.error('demo is the fixed teaching example; use inspect/generate for a custom corpus')
        result = demo()
    else:
        corpus = json.loads(args.corpus.read_text(encoding='utf-8')) if args.corpus else CORPUS
        counts = train(corpus)
        if args.command == 'inspect':
            result = {'context': args.context, 'counts': counts.get(args.context, {}),
                      'probabilities': probabilities(counts, args.context)}
        else:
            result = generate(counts, args.prompt, args.length, args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
