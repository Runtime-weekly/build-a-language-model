# Vectors, weights, bias and batching

These are the exact small calculations used in RUNTIME's learning lessons.
They illustrate the parts of a linear layer. The weights are chosen by hand;
this example does not train a model or measure language quality.

## Run

Python 3.10 or later, standard library only. From this directory:

```sh
python3 -B -m unittest -v test_math
python3 -B -c "from test_math import example; import json; print(json.dumps(example(), indent=2))"
```

Expected: nine passing tests. Tested on Python 3.12.3, Linux ARM64; other
platforms have not been tested here. Use `python` if that is your Python 3
command. Run in this directory so `math_model` can be imported.

## Follow the numbers

Count `a`, `t`, and spaces in a three-character window, in that order.
`aat` gives `x = [2, 1, 0]`; `a t` gives `[1, 1, 1]`.
These designed count features are not learned embeddings. `ata` and `aat`
produce the same vector because counts discard character order.

Use a row-vector convention. The matrix has one row per input feature and one
column per output:

```text
W = [[ 2, -1],
     [-1,  2],
     [ 1,  0]]
b = [1, -1]
```

For the first column, multiply matching entries: `2*2 + 1*(-1) + 0*1 = 3`.
For the second: `2*(-1) + 1*2 + 0*0 = 0`.
Add the bias: `xW + b = [4, -1]`. These are raw scores, not probabilities.
With zero inputs, the output remains `b`.

Batch the two windows into a `2 x 3` matrix. Multiplying by the `3 x 2`
weights gives a `2 x 2` result. Add the same bias to each row:

```text
[[2, 1, 0],      [[4, -1],
 [1, 1, 1]]  →   [3,  0]]
```

The rows are independent examples. This operation does not let tokens attend
to each other. A nonzero bias makes the map affine, although common software
calls the component a linear layer. Some libraries store transposed weights;
the tests explicitly check that convention.

## Try and check

1. Change only the bias. Which outputs move, and by how much?
2. Change only the second batch input. Does the first output change?
3. Predict what happens when the space-count input is zero.
4. Pass incompatible shapes. The implementation should reject them.

`[3,1,0]` is included only as an arithmetic probe in the test data: its counts
sum to four, so it cannot describe a three-character window.

For the prediction/counting baseline and course introduction, start with the
[course overview](../../README.md) and [Lesson 1](../01/README.md).

