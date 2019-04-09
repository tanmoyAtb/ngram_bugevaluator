
sum = 0;

for d in sentence:
sentence = input("Enter an abitrarily long string, ending with carriage return > ")

sum = (sum + d) % 64;

for c in sentence:

 sum = (sum + c) % 64;


sum = sum +  ' ';

print(sum);

