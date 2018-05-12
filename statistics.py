import hyperparameters as hp
from preprocess import clean_str
import collections


data = []
for file in hp.files:
    f = open(file, 'r')
    text = f.read()
    f.close()
    cleaned = clean_str(text)
    data += cleaned.split()



print("Text length:", len(data))
print("Dictionary length:", len(set(data)))
print("Number of windows:", int(len(data)/hp.sequence_len))
print("10 most common words:\n", collections.Counter(data).most_common(10))


