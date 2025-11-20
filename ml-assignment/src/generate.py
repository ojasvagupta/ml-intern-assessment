  # -------------------------
    # Generate text
    # -------------------------
    def generate(self, max_tokens: int = 50, seed: int = None) -> str:
        """Generate a sentence (string) using the trained model."""
        if seed is not None:
            random.seed(seed)

        # start history with n-1 start tokens
        history = tuple(["<s>"] * (self.n - 1))
        output_tokens = []

        for _ in range(max_tokens):
            # try full n-gram distribution
            dist = self.probs.get(history, None)

            # backoff strategy if history unseen:
            if dist is None:
                # try backoff to bigram (last token of history)
                if self.n >= 2:
                    bigram_hist = (history[-1],)
                    dist = self.bigram_prob.get(bigram_hist, None)
                # if still None, use unigram
                if dist is None:
                    dist = self.unigram_prob

            # sample next token
            nxt = self._sample_from_dist(dist)

            if nxt == "</s>":
                break
            output_tokens.append(nxt)

            # shift history window
            history = tuple(list(history[1:]) + [nxt]) if self.n > 1 else tuple()

        return " ".join(output_tokens)
