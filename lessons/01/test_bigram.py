import copy
import unittest
from bigram import CORPUS, shifted, train, probabilities, generate, heldout_score, vocabulary, encode, decode


class BigramTests(unittest.TestCase):
    def test_vocabulary_encoding_roundtrip(self):
        tokens = vocabulary(CORPUS)
        self.assertEqual(tokens, sorted(set(''.join(CORPUS))))
        self.assertEqual(tokens, vocabulary(list(reversed(CORPUS))))
        for text in CORPUS + ['', 'cat']:
            self.assertEqual(decode(encode(text, tokens), tokens), text)

    def test_encoding_rejects_unknown_characters_and_invalid_ids(self):
        tokens = vocabulary(CORPUS)
        for call in [lambda: encode('?', tokens), lambda: decode([-1], tokens),
                     lambda: decode([len(tokens)], tokens), lambda: decode([True], tokens),
                     lambda: decode([1.5], tokens), lambda: decode(['1'], tokens),
                     lambda: encode('a', ['a', 'a']), lambda: encode('a', ['ab'])]:
            with self.assertRaises(ValueError):
                call()

    def test_shifted_inputs_targets(self):
        self.assertEqual(shifted('cat'), {'inputs': ['c', 'a'], 'targets': ['a', 't']})
        self.assertEqual(shifted(''), {'inputs': [], 'targets': []})
        self.assertEqual(shifted('x'), {'inputs': [], 'targets': []})

    def test_exact_counts_and_total(self):
        counts = train(CORPUS)
        self.assertEqual(sum(sum(row.values()) for row in counts.values()), 33)
        self.assertEqual(counts['t'], {' ': 2, '.': 2, 'h': 3})
        self.assertEqual(counts['a'], {'n': 1, 't': 4})
        self.assertEqual(counts[' '], {'c': 2, 'd': 1, 'r': 1, 's': 2})

    def test_probability_rows(self):
        counts = train(CORPUS)
        self.assertEqual(probabilities(counts, 't'), {' ': 2/7, '.': 2/7, 'h': 3/7})
        for context, row in counts.items():
            probs = probabilities(counts, context)
            self.assertAlmostEqual(sum(probs.values()), 1)
            self.assertTrue(all(0 < p <= 1 for p in probs.values()))

    def test_document_boundaries_not_counted(self):
        self.assertEqual(train(['ab', 'cd']), {'a': {'b': 1}, 'c': {'d': 1}})
        self.assertNotIn('.', train(CORPUS))

    def test_seeded_sampling_and_trace(self):
        counts = train(CORPUS)
        before = copy.deepcopy(counts)
        a = generate(counts, seed=7)
        self.assertEqual(a, generate(counts, seed=7))
        self.assertEqual(counts, before)
        text = 't'
        for step in a['steps']:
            self.assertEqual(step['context'], text[-1])
            row = counts[step['context']]
            ticket = step['ticket']
            for token, count in sorted(row.items()):
                if ticket < count:
                    self.assertEqual(step['next_character'], token)
                    break
                ticket -= count
            text += step['next_character']
        self.assertEqual(text, a['text'])

    def test_only_last_character_is_context(self):
        counts = train(CORPUS)
        a = generate(counts, prompt='t')
        b = generate(counts, prompt='unrelated prefix t')
        self.assertEqual([(s['context'], s['next_character'], s['ticket']) for s in a['steps']],
                         [(s['context'], s['next_character'], s['ticket']) for s in b['steps']])

    def test_unseen_context_and_length_limit(self):
        counts = train(CORPUS)
        for context in ['?', '.']:
            self.assertEqual(probabilities(counts, context), {})
            result = generate(counts, prompt=context)
            self.assertEqual(result['stop_reason'], 'period' if context == '.' else 'unseen_context')
            self.assertEqual(result['generated_characters'], 0)
        self.assertEqual(generate(train(['aaaa']), max_new_characters=5, prompt='a')['text'], 'aaaaaa')
        self.assertEqual(generate(counts, max_new_characters=0)['text'], 't')

    def test_period_is_explicit_stop_even_with_outgoing_counts(self):
        result = generate(train(['a.b']), prompt='a', max_new_characters=1)
        self.assertEqual(result['text'], 'a.')
        self.assertEqual(result['stop_reason'], 'period')
        self.assertEqual(generate(train(['a.b']), prompt='.', max_new_characters=3)['text'], '.')

    def test_heldout_no_leakage_zero_probability_is_infinite(self):
        before = copy.deepcopy(CORPUS)
        score = heldout_score(CORPUS[:2], CORPUS[2:])
        self.assertEqual(CORPUS, before)
        self.assertEqual(score['pair_count'], 11)
        self.assertEqual(score['status'], 'infinite_unsmoothed_loss')
        self.assertIsNone(score['mean_nll_nats'])
        self.assertEqual([(r['context'], r['target']) for r in score['zero_probability_pairs']],
                         [(' ', 'r'), ('r', 'a'), ('a', 'n'), ('n', '.')])
        self.assertEqual(heldout_score(['abab'], ['ab'])['mean_nll_nats'], 0)
        with self.assertRaises(ValueError):
            heldout_score(['ab'], ['ab'])

    def test_invalid_inputs(self):
        for call in [lambda: train([]), lambda: train(['ok', 1]),
                     lambda: probabilities({}, 'two'), lambda: generate({}, prompt=''),
                     lambda: generate({}, max_new_characters=-1)]:
            with self.assertRaises(ValueError):
                call()


if __name__ == '__main__':
    unittest.main()
