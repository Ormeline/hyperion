import spacy

nlp = spacy.load("en_core_web_sm")

# Define the words to compare
word1 = "cat"
word2 = "monkey"
word3 = "banana"

# Calculate the similarity between the words using spaCy
similarity1 = nlp(word1).similarity(nlp(word2))
similarity2 = nlp(word3).similarity(nlp(word2))
similarity3 = nlp(word3).similarity(nlp(word1))

# Print the similarities
print(f"Similarity between '{word1}' and '{word2}': {similarity1}")
print(f"Similarity between '{word3}' and '{word2}': {similarity2}")
print(f"Similarity between '{word3}' and '{word1}': {similarity3}")

# Define the tokens to compare
tokens = "cat apple monkey banana"
tokens = nlp(tokens)

# Calculate the similarities between the tokens using spaCy
for token1 in tokens:
    for token2 in tokens:
        similarity = token1.similarity(token2)
        print(f"Similarity between '{token1.text}' and '{token2.text}': {similarity}")

# Define the sentence to compare
sentence_to_compare = "Why is my cat on the car"

# Define the sentences to compare against
sentences = [
    "Where did my dog go",
    "Hello, there is my car",
    "I've lost my car in my car",
    "I'd like my boat back",
    "I will name my dog Diana",
]

# Calculate the similarities between the sentence and each sentence in the list using spaCy
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = model_sentence.similarity(nlp(sentence))
    print(f"Similarity between '{sentence}' and '{sentence_to_compare}': {similarity}")

# It's interesting to see that the similarity between 'cat' and 'monkey' is higher than the similarity between 'banana' and 'monkey'. 
# This could be because 'cat' and 'monkey' are both animals, while 'banana' is a fruit, which is a less similar category. 
# This highlights how word embeddings can capture subtle relationships between words based on their context.
# Add another example to compare using spaCy
example_sentence = "The quick brown fox jumped over the lazy dog"
example_sentences = [
    "The red fox is running in the field",
    "The lazy dog is sleeping on the couch",
    "The brown bear is fishing in the river",
    "The black cat is sitting on the window sill",
    "The white rabbit is eating carrots in the garden"
]

model_sentence = nlp(example_sentence)

for sentence in example_sentences:
    similarity = model_sentence.similarity(nlp(sentence))
    print(f"Similarity between '{sentence}' and '{example_sentence}': {similarity}")