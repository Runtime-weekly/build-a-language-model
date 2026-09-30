# Character counts before transformers

This is a deliberately small character **bigram** baseline. Each character
predicts the next character using counts from an authored corpus:

```text
the cat sat.
the dog sat.
the cat ran.
```

Each document contains 12 characters, including spaces and the period, giving
11 adjacent pairs and 33 pairs total. No pair crosses a document boundary.
For `cat`, shifted inputs are `c,a` and targets are `a,t`. The count model uses characters directly. For the later coding lesson,
`vocabulary(documents)` makes a sorted list of unique characters, `encode(text,
tokens)` maps them to zero-based IDs, and `decode(ids,tokens)` reconstructs text.
Unknown characters and invalid IDs raise errors; no unknown-token substitution
is hidden. The encoding demo is in `demo.json`.

The row after `a` has `t:4, n:1`, so its probabilities are **80% t, 20% n**.
The row after `t` has space:2, period:2, h:3: probabilities 2/7, 2/7 and 3/7.
The model has no knowledge of words, meaning or sentence structure. Even a long
prompt conditions its next prediction only on the last character.

## Run it

Python 3.10+; standard library only. From this folder:

```sh
python3 -B -m unittest -v test_bigram
python3 -B bigram.py demo
python3 -B bigram.py inspect --context a
python3 -B bigram.py generate --prompt ca --seed 0 --length 6
python3 -B bigram.py generate --prompt '?' --seed 7
```

`demo.json` is the saved output of `python3 -B bigram.py demo > demo.json`.
It contains all counts and probability rows, shifted documents, seeded traces,
an unseen-context example and a separate heldout-document calculation.
The displayed row-a seed 0 was selected to provide six visible steps; it is not
a quality comparison. Other seeds can stop earlier. Trace entries contain
prefixes before/after, sampled character, count, probabilities
and the exact integer random ticket. Outcomes can end early; a six-character
limit is not a promise to generate six characters. Tested on Python 3.12.3 on Linux ARM64. Repeatability is within the tested Python
implementation, not a guarantee about arbitrary future random-number APIs.

For your own corpus, save a UTF-8 JSON list of document strings and pass
`--corpus my-corpus.json` to `inspect` or `generate`. Empty or one-character
documents contribute no adjacent pairs. No packages, network, model download,
GPU, server or credentials are needed.

## What happens with missing data?

There is no smoothing. Generation explicitly stops on a period (`period`),
at its length cap (`length_limit`), or when the last character has no outgoing
observations (`unseen_context`). Period stopping is a teaching rule, not a learned
end-of-sequence token; it applies even in a custom corpus containing a transition
after a period. A zero-character budget returns the prompt unchanged with
`length_limit`. Missing rows return an empty distribution, never a uniform fallback.

The demo generation fits all three documents. **Separately**, heldout scoring
fits the first two and scores the third. It does not train on heldout pairs.
Four transitions in the third document have zero training probability, so
unsmoothed negative log likelihood is infinite. JSON reports
`status: infinite_unsmoothed_loss` and `mean_nll_nats: null`, rather than a
misleading finite score. A split with no pairs is explicitly `no_pairs`.
No accuracy or generalization claim is based on this tiny split.

## Scope

This is counting and normalization, not a transformer, neural network, embedding
model or gradient-based training. Sampling is not choosing the highest-probability
character every time. Counts stay unchanged during generation. The code is an
Episode 1A arithmetic aid and a small reproducible starting point for 1B.
