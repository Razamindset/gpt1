from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        self.trained = False

    def train(self, text, num_merges=500):
        # Count words
        word_freq = Counter(text.split())

        # Character vocabulary
        vocab = Counter(
            {tuple(word): freq for word, freq in word_freq.items()}
        )

        # Learn merges
        for _ in range(num_merges):
            pairs = self._get_pair_frequencies(vocab)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # IMPORTANT
            vocab = self._merge_pair(vocab, best_pair)

        self._build_vocab(word_freq)
        self.trained = True

    def encode(self, text):
        tokens = []

        for word in text.split():
            symbols = list(word)

            for pair in self.merges:
                symbols = self._merge_symbols(symbols, pair)

            tokens.extend(self.token_to_id[s] for s in symbols)

        return tokens

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        return "".join(tokens)

    def _merge_pair(self, vocab, pair):
        new_vocab = Counter()

        for word, freq in vocab.items():
            new_word = []

            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_vocab[tuple(new_word)] += freq

        return new_vocab

    def _merge_symbols(self, symbols, pair):
        merged = []

        i = 0
        while i < len(symbols):
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1

        return merged

    def _get_pair_frequencies(self, vocab):
        pairs = Counter()

        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq

        return pairs

    def _build_vocab(self, word_freq):
        tokens = set()

        for word in word_freq:
            symbols = list(word)

            for pair in self.merges:
                symbols = self._merge_symbols(symbols, pair)

            tokens.update(symbols)

        self.token_to_id = {
            token: i
            for i, token in enumerate(sorted(tokens))
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }