# Build a language model with RUNTIME

A practical course that pairs an animated explanation with a Python implementation.
Start with a transparent character-count baseline, then build the components of a small transformer.

## Available now

- **[Watch 1A: What are we actually teaching a language model?](https://youtu.be/hSht0qBgSaE)** The next-token task, character tokens, shifted targets, counts, probabilities and the limits of one-character context.
- [Lesson 1 code and exercises](lessons/01/README.md): a runnable standard-library baseline. The companion 1B video is in preparation.
- [Lesson 2 calculations and tests](lessons/02/README.md): vectors, dot products, bias, matrix outputs and independent batch rows.

The learning Shorts use the same examples, one narrow topic at a time:
next-token prediction; token IDs; shifted targets; pair counts; row
probabilities; sampling; context limits; vectors; dot products; bias;
matrix outputs; and batch shapes. Videos are available only when linked;
this list is the sequence, not a claim that every Short has been released.

This first model is **not a transformer or neural network**. Its probabilities come from adjacent-character counts in three original sentences. The corpus is small enough to inspect completely, not a benchmark for language ability.

## Run the first lesson

Install Python 3.10 or newer. No packages, accounts, GPU or API key are needed.
Download this repository, or clone it:

```sh
git clone https://github.com/Runtime-weekly/build-a-language-model.git
cd build-a-language-model/lessons/01
python3 -B -m unittest -v test_bigram
python3 -B bigram.py inspect --context a
python3 -B bigram.py generate --prompt ca --seed 0 --length 6
```

Expected: twelve passing tests; after `a`, counts `n:1, t:4`, giving probabilities `0.2, 0.8`. The seeded six-step example produces `cathe do` and stops at the length limit. The seed illustrates the loop; it was not chosen as evidence of quality.

Tested with Python 3.12.3 on Linux ARM64. Other platforms have not been tested here. If your system uses `python` instead of `python3`, use that command after checking its version.

## Course route

Each number pairs **A: understand** with **B: build**. Subsequent videos are planned unless linked as available.

1. Next-token prediction and a character-count baseline.
2. Vectors, weights, biases and linear layers.
3. Loss, gradients and learning.
4. Nonlinearities and feedforward networks.
5. Text embeddings and position.
6. One attention head.
7. Causal masks and multiple heads.
8. Residual connections, normalization and a decoder block.
9. Assemble, train and save a small transformer.
10. Autoregressive generation and sampling choices.
11. Tokenization, held-out evaluation and honest comparisons.
12. Batching, precision, memory and caching.
13. What changes at large scale: parameter budgets, expert routing and distributed execution.

## Check understanding

- For `dog.`, write the input and target lists. Why are they shifted?
- Why do `the cat sa` and `the cat ra` receive the same next-character distribution here?
- Could five independent draws all be `t` when its probability is 0.8?
- Add an original sentence to the corpus and predict a changed count before running it.

Track watching, understanding, implementing and verifying separately. Running the tests does not prove comprehension or useful generalization.

## Sources and limitations

The count estimator follows [Jurafsky and Martin, N-gram Language Models, equation 3.10](https://web.stanford.edu/~jurafsky/slp3/3.pdf), applied to characters.
There is no smoothing. Unseen transitions have zero probability; a missing outgoing row stops generation. The separate held-out demonstration therefore reports infinite unsmoothed loss rather than hiding unseen pairs with a finite score.

Questions, reproducible corrections and different experiments are welcome through issues or pull requests. Include your command, data and expected versus observed result. Do not include credentials or private data.

The original code and tutorial text are MIT licensed. The license does not grant rights to RUNTIME branding, video or voice recordings, which are not included in this repository.
