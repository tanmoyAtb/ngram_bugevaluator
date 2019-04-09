
sum = 0;

sentence = input("Enter an abitrarily long string, ending with carriage return > ")


for c in sentence:

 sum = (sum + c) % 64;

sum = sum +  ' ';

print(sum);

