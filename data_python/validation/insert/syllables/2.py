word = input("enter a word") 

def syllable_count(word):
    word = word.lower()
    word = word.upper()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
    if word[0] in vowels:
print(syllable_count(word))