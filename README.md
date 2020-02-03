# MultiVerse: Probabilistic Programming Language for Causal Reasoning

This is a basic prototype implementation of probabilistic programming MultiVerse for performing counterfactual inference using importance sampling.


# Paper

`MultiVerse: Causal Reasoning using Importance Sampling in Probabilistic Programming` by Yura Perov (equal contribution), Logan Graham (equal contribution), Kostis Gourgoulias, Jonathan G. Richens, Ciarán M. Lee, Adam Baker, Saurabh Johri.

Accepted to the 2nd Symposium on Advances in Approximate Bayesian Inference (2019) and to the Second International Conference on Probabilistic Programming (2020).

URL: https://arxiv.org/abs/1910.08091

## Citation

A suggested BibTeX citation:

```
@misc{babylon2019multiverse,
  title={Multi{V}erse: {C}ausal reasoning using importance sampling in probabilistic programming},
  author={Perov, Yura and Graham, Logan and Gourgoulias, Kostis and Richens, Jonathan G. and Lee, Ciar{\'a}n M. and Baker, Adam and Johri, Saurabh},
  url={https://arxiv.org/abs/1910.08091},
  year={2019},
  note={Logan and Yura have contributed equally. Accepted to the 2nd Symposium on Advances in Approximate Bayesian Inference (2019) and to the Second International Conference on Probabilistic Programming (2020).}
}
```


# Tests

```
python -m pytest tests/
```

Basic code formatting checks:
`make lint`


# Copyright and Licence

Copyright 2019-2020 Babylon Health (Babylon Partners Limited).

MIT Licence.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
