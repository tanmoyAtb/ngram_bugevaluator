
sum = 0;

for c in sentence:
sentence = input("Enter an abitrarily long string, ending with carriage return > ")

sum = (sum + c) % 64;
print(sum);

for c in sentence:

 sum = (sum + c) % 64;


sum = sum +  ' ';

print(sum);

