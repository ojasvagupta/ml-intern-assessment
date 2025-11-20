# evaluation.md

## Task
Build and evaluate a trigram (N=3) language model from scratch that supports cleaning, padding, unknown words, and probabilistic generation.

## Design choices
1. **Preprocessing**: Lowercased text, removed non-alphanumeric noise with a regex, split sentences by `.!?` and tokenized by whitespace and simple punctuation rules. Chosen for reproducibility and simplicity suitable for Project Gutenberg texts.

2. **Padding**: Each sentence is padded with `n-1` start tokens `<s>` and one end token `</s>` so initial contexts (e.g., sentence beginnings) are modeled properly.

3. **Unknown handling**: Words with frequency ≤ `unk_threshold` are replaced by `<UNK>`. This prevents the model encountering unseen tokens during generation and stabilizes probability mass distribution.

4. **Data structures**: A nested dictionary (`counts[history_tuple][next_word] = count`) stores n-gram counts. Totals per-history are stored separately to compute probabilities efficiently.

5. **Smoothing**: Add-k (Lidstone) smoothing is applied when converting counts to probabilities. Default `k=1e-6` (tiny) to keep distribution similar to raw counts but avoid exact zeros. For stronger smoothing, set `k=1.0` (add-one).

6. **Backoff strategy**: If a full-order history is unseen, back off to lower orders: trigram → bigram → unigram. This is a pragmatic fallback to allow generation even with sparse histories.

7. **Generation**: Probabilistic sampling uses `random.choices` over the probability distribution for the current history, not greedy argmax. This creates realistic (and varied) text.

## Limitations & future improvements
- Tokenization is simple; better tokenization (e.g., SpaCy) would handle contractions and punctuation more accurately.
- Backoff implemented simply; advanced smoothing/backoff (Katz, Kneser-Ney) would improve perplexity.
- The model currently treats punctuation tokens as words; for downstream use you may want separate handling.
- For larger corpora, batching and more memory-efficient structures (sparse matrices) would scale better.

## How to run
- Place a Project Gutenberg plain text file (e.g., `alice_in_wonderland.txt`) in the same folder.
- Run `python trigram_model.py`. Edit `unk_threshold`, `smoothing_k`, or `n` if experimenting.


