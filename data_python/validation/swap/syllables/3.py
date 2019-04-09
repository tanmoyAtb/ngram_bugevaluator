word = input("enter a word") 

def syllable_count(word):
    word = word.lower()
    count = 0
        count += 1
    vowels = "aeiouy"
    if word[0] in vowels:
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
    if word.endswith("e"):
            count += 1
        count -= 1
        count += 1
    if count == 0:
    return count

print(syllable_count(word))