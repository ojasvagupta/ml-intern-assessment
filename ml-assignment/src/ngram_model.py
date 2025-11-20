import re
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Iterable

class NGramLanguageModel:
    def __init__(self, n: int = 3, unk_threshold: int = 1, smoothing_k: float = 1e-6):
        assert n >= 1
        self.n = n
        self.unk_threshold = unk_threshold
        self.smoothing_k = smoothing_k
        # counts: history_tuple -> {next_word: count}
        self.counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # totals: history_tuple -> total count
        self.totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.vocab: set = set()
        self.unigram_counts: Counter = Counter()
        self.vocab_size = 0
        # probability distributions
        self.probs: Dict[Tuple[str, ...], Dict[str, float]] = {}

    # -------------------------
    # Preprocessing helpers
    # -------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        # Lowercase and normalize whitespace. Keep basic punctuation used to split sentences.
        text = text.lower()
        # Replace non-ascii with space, keep basic punctuation .?!'-,() for simple tokenization
        text = re.sub(r"[^a-z0-9\.\!\?\,\;\:\'\"\-\(\)\s]", " ", text)
        # collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Very simple sentence splitter by .!? (works for classics reasonably well)
        sentences = re.split(r'(?<=[\.!\?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    @staticmethod
    def _tokenize(sentence: str) -> List[str]:
        # Tokenize on whitespace and simple punctuation boundaries
        # Keep apostrophes inside words (e.g., don't -> don't)
        # Separate commas and parentheses
        sentence = sentence.replace("(", " ( ").replace(")", " ) ")
        # ensure punctuation tokens are separated
        sentence = re.sub(r'([,\.\!\?\;\:])', r' \1 ', sentence)
        tokens = sentence.split()
        return tokens

    def _pad_tokens(self, tokens: List[str]) -> List[str]:
        # pad with n-1 start tokens and one end token
        start = ["<s>"] * (self.n - 1)
        end = ["</s>"]
        return start + tokens + end

    # -------------------------
    # Training / fitting
    # -------------------------
    def fit(self, raw_text: str):
        """Train the n-gram model from raw text."""
        cleaned = self._clean_text(raw_text)
        sentences = self._split_sentences(cleaned)

        # First pass: collect word frequencies (unigrams)
        word_counts = Counter()
        all_sent_tokens = []
        for sent in sentences:
            tokens = self._tokenize(sent)
            padded = self._pad_tokens(tokens)
            all_sent_tokens.append(padded)
            # count tokens except the start tokens for unigram frequency
            for t in padded[(self.n - 1):]:  # skip the padding when counting freq
                word_counts[t] += 1

        # Build vocabulary: words with freq > unk_threshold
        self.vocab = {w for w, c in word_counts.items() if c > self.unk_threshold}
        self.vocab.update({"<s>", "</s>", "<UNK>"})  # ensure specials exist
        self.vocab_size = len(self.vocab)

        # Second pass: replace rare words with <UNK>, and build n-gram counts
        self.unigram_counts = Counter()
        for padded in all_sent_tokens:
            # replace rare words
            tokens = [
                t if (t in self.vocab and t not in ("<s>", "</s>")) else ("<UNK>" if t not in ("<s>", "</s>") else t)
                for t in padded
            ]
            # update unigram counts (skip start padding or include depending on use; include for modeling)
            for t in tokens:
                self.unigram_counts[t] += 1

            # build counts for all n-gram histories in this sentence
            for i in range(self.n - 1, len(tokens)):
                history = tuple(tokens[i - (self.n - 1): i])  # (n-1) history
                target = tokens[i]
                self.counts[history][target] += 1
                self.totals[history] += 1

        # Convert counts to probabilities (with smoothing)
        self._build_probabilities()

    def _build_probabilities(self):
        """Convert counts to probability distributions with add-k smoothing."""
        self.probs = {}
        V = self.vocab_size
        for history, nexts in self.counts.items():
            total = self.totals[history]
            denom = total + self.smoothing_k * V
            dist = {}
            # include all vocabulary words in distribution (so unseen words get non-zero prob after smoothing)
            for w in self.vocab:
                c = nexts.get(w, 0)
                dist[w] = (c + self.smoothing_k) / denom
            # normalize (should already sum to 1 but numerical safety)
            s = sum(dist.values())
            for w in dist:
                dist[w] /= s
            self.probs[history] = dist

        # Also create lower-order distributions for backoff (bigram, unigram). Simple approach:
        # unigram dist:
        total_unigrams = sum(self.unigram_counts.values())
        self.unigram_prob = {w: (self.unigram_counts.get(w, 0) + self.smoothing_k) / (total_unigrams + self.smoothing_k * self.vocab_size)
                             for w in self.vocab}

        # bigram dist if applicable
        if self.n >= 2:
            # build bigram counts from trigram counts aggregated by last token of history
            self.bigram_counts = defaultdict(lambda: defaultdict(int))
            self.bigram_totals = defaultdict(int)
            for history, nexts in self.counts.items():
                # history is length n-1. For trigram (n=3), history is (w1,w2)
                # bigram history = last token of trigram history -> (w2,)
                bigram_history = (history[-1],)
                for nxt, c in nexts.items():
                    self.bigram_counts[bigram_history][nxt] += c
                    self.bigram_totals[bigram_history] += c
            self.bigram_prob = {}
            for h, nexts in self.bigram_counts.items():
                denom = self.bigram_totals[h] + self.smoothing_k * self.vocab_size
                dist = {}
                for w in self.vocab:
                    c = nexts.get(w, 0)
                    dist[w] = (c + self.smoothing_k) / denom
                s = sum(dist.values())
                for w in dist:
                    dist[w] /= s
                self.bigram_prob[h] = dist

    # -------------------------
    # Utility: choose next token by distribution
    # -------------------------
    @staticmethod
    def _sample_from_dist(dist: Dict[str, float]) -> str:
        words = list(dist.keys())
        probs = list(dist.values())
        # random.choices samples with weights; it returns a list
        chosen = random.choices(words, weights=probs, k=1)[0]
        return chosen
