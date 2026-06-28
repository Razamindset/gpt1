from collections import Counter
from itertools import islice

# Start by reading the file 
with open("input.txt", encoding="utf-8") as f:
    text = f.read()

# print(f"Characters : {len(text)}")
# print(f"Words      : {len(text.split())}")
# print(f"Lines      : {len(text.splitlines())}")

# # Unique chars
# characters = sorted(set(text))
# print("Unique chars: ", characters)
# print(len(characters))


# returns a dict of string: number
word_freq = Counter(text.split())

vocab = {}
# Now we need to  convert every word into chars 
for word, freq in word_freq.items():
    vocab[tuple(word)] = freq

def print_vocab(vocab, limit=10):
    for word, freq in islice(vocab.items(), limit):
        print(word, "->", freq)

# Lets create pair freq
def get_pair_frequencies(vocab):
    pair_freq = Counter()

    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq

    return pair_freq

pairs = get_pair_frequencies(vocab)

# print(pairs.most_common(10))

# this fucntion will merge the most common pairs in the vocab into one unit 
def merge_pair(vocab: dict, pair):
    new_vocab = Counter() 

    for word, freq in vocab.items():
        new_word = []

        i = 0

        # we loop over the word
        while i < len(word):
            # now we will check if our pair exists in the word
            # if it exits then we replace those chars

            if(
                i < len(word) -1
                and word[i] == pair[0]
                and word[i+1] == pair[1]
            ):
                new_word.append(word[i] + word[i+1])
                i+=2
            else:
                # No change
                new_word.append(word[i])
                i+=1

        new_vocab[tuple(new_word)] += freq

    return new_vocab


num_merges = 1000
merges = []

for i in range(num_merges):

    # Count all adjacent pairs
    pairs = get_pair_frequencies(vocab)

    # Stop if there are no pairs left
    if not pairs:
        break

    # Find the most common pair
    best_pair, freq = pairs.most_common(1)[0]

    print(f"Merge {i+1}: {best_pair} ({freq} occurrences)")

    # Remember this merge
    merges.append(best_pair)

    # Update the vocabulary
    vocab = merge_pair(vocab, best_pair)

print("\nLearned merges:")
for merge in merges[:20]:
    print(merge)


def merge_symbols(symbols, pair):
    new_symbols  = []

    i = 0

    while i < len(symbols):
        if(
            i < len(symbols) -1
            and symbols[i] == pair[0]
            and symbols[i + 1] == pair[1]
            ):
            new_symbols.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1

    return new_symbols


def encode_word(word, merges):
    symbols = list(word)

    for pair in merges:
        symbols = merge_symbols(symbols, pair)

    return symbols

print(encode_word("understanding", merges))

# Create tokens 
all_tokens = set()

for word in word_freq:
    tokens = encode_word(word, merges)

    for token in tokens:
        all_tokens.add(token)

sorted(all_tokens)

token_to_id = {}

for i, token in enumerate(sorted(all_tokens)):
    token_to_id[token] = i


print(list(all_tokens)[:10])