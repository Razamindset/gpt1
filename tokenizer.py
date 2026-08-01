from collections import Counter, defaultdict
import json

class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}

        self.vocab = Counter()

        self.trained = False

        self.special_tokens = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ]

    
    def train(self, text, num_merges=669): 
        # returns a dict of string: number
        words = []

        for i, word in enumerate(text.split()):
            if i == 0:
                words.append(word)
            else:
                words.append("_"+word)


        word_freq = Counter(words)
        # vocab: word-as-tuple-of-symbols -> frequency
        vocab = {tuple(word): freq for word, freq in word_freq.items()}

        # Example vocab at this point 
        # vocab = { ('q', 'w', 'e', 'r', 't', 'y'): count }

        # --- Incremental pair-frequency index ---
        # The naive approach recomputes pair counts over the ENTIRE vocab on
        # every merge (O(num_merges * vocab_size)), which is what made this
        # slow on real corpora. Instead we keep running pair counts and, for
        # each merge, only touch the (typically small) set of words that
        # actually contain the pair being merged.
        pair_freq = Counter()
        pair_to_words = defaultdict(set)

        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += freq
                pair_to_words[pair].add(word)

        try:
            from tqdm.auto import tqdm
            merge_range = tqdm(range(num_merges), desc="BPE merges")
        except ImportError:
            merge_range = range(num_merges)

        for _ in merge_range:
            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            if pair_freq[best_pair] <= 0:
                break

            self.merges.append(best_pair)

            # Only the words that contain best_pair need updating.
            affected_words = list(pair_to_words.get(best_pair, ()))

            for old_word in affected_words:
                freq = vocab.get(old_word)
                if freq is None:
                    continue

                # Remove this word's contribution to every pair it forms.
                for i in range(len(old_word) - 1):
                    p = (old_word[i], old_word[i + 1])
                    pair_freq[p] -= freq
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                    bucket = pair_to_words.get(p)
                    if bucket is not None:
                        bucket.discard(old_word)
                        if not bucket:
                            del pair_to_words[p]

                new_word = tuple(self._merge_symbols(list(old_word), best_pair))

                del vocab[old_word]
                vocab[new_word] = vocab.get(new_word, 0) + freq

                # Add the merged word's contribution to every pair it forms.
                for i in range(len(new_word) - 1):
                    p = (new_word[i], new_word[i + 1])
                    pair_freq[p] += freq
                    pair_to_words[p].add(new_word)

            pair_freq.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

        # Training is finished
        # We learnt some merges in the loop
        # now we want to use them
        # Apply the merges over the words freq and save unique tokens
        self._build_vocab(word_freq)
        self.vocab = Counter(vocab)
        self.trained = True

    def encode(self, text, add_special_tokens=True):
        # Suppose we now have some new text that we want to encode using the learnt merges 
        # this funciton is for that 

        if not self.trained:
            raise RuntimeError("Tokenizer has not been trained.")

        tokens = []

        if add_special_tokens:
            tokens.append(self.token_to_id['<bos>'])

        words = text.split()

        for i, word in enumerate(words):
            if i != 0:
                word = "_" + word

            symbols = list(word)

            for pair in self.merges:
                symbols = self._merge_symbols(symbols, pair)
            
            # Handle if there is some unknown token
            unk_id  = self.token_to_id['<unk>']

            tokens.extend(
                self.token_to_id.get(symbol, unk_id)
                for symbol in symbols
            )
        
        if add_special_tokens:
            tokens.append(self.token_to_id['<eos>'])

        return tokens

    def decode(self, ids, skip_special_tokens=True):
        tokens = []

        unk_token = "<unk>"

        for idx in ids:
            token = self.id_to_token.get(idx, "<unk>")

            if skip_special_tokens == True and token in self.special_tokens:
                continue

            tokens.append(token)

        text = "".join(tokens)

        return text.replace("_", " ")

    def _merge_symbols(self, symbols, pair):
        # we get a list of symbols for a word and a possible merge  
        # if we can make a similar pair from the list of availble items we merge it and make new symbols
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

    def _build_vocab(self, word_freq):
        tokens = set()
        
        for word in word_freq:
            symbols = list(word)

            for pair in self.merges:
                # Get new symbols if anyt merged 
                symbols = self._merge_symbols(symbols, pair)

            tokens.update(symbols)

        self.token_to_id = {}
        # First add the special tokens
        for token in self.special_tokens:
            self.token_to_id[token] = len(self.token_to_id)
        
        # Now add the learnt tokens 
        for token in sorted(tokens):
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        # Dict for id to token 
        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

    def save(self, path):
        data = {
            'merges': [list(pair) for pair in self.merges],
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.merges = [
            tuple(pair)
            for pair in data['merges']
        ]


        self.token_to_id = data["token_to_id"]

        self.special_tokens = data["special_tokens"]

        # remake id to token using token to id 
        self.id_to_token = {
            idx: token
            for token, idx in self.token_to_id.items()
        }

        self.trained = True 

# # Start by reading the file 
# with open("input.txt", encoding="utf-8") as f:
#     text = f.read()

# tokenizer = BPETokenizer()

# print("Training...")
# tokenizer.train(text, num_merges=100)

# print("Training complete!")
# print()

# print("First 20 merges:")
# for merge in tokenizer.merges[:20]:
#     print(merge)

# print("\nVocabulary size:", len(tokenizer.token_to_id))

# print("\nFirst 20 tokens:")
# for token, idx in list(tokenizer.token_to_id.items())[:20]:
#     print(idx, repr(token))

# sentence = "The quick brown fox"

# ids = tokenizer.encode(sentence)

# print("\nEncoded IDs:")
# print(ids)

# decoded = tokenizer.decode(ids)

# print("\nDecoded:")
# print(repr(decoded))

# tokenizer.save("tokenizer.json")


# new_tokenizer.load("tokenizer.json")

# ids = new_tokenizer.encode("The quick brown fox")

# print(ids)

# print(new_tokenizer.decode(ids))