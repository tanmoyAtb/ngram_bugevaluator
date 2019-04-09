
sentence = input("Enter an abitrarily long string, ending with carriage return > ")
sum = 0;



for c in sentence:

 sum = (sum + c) % 64;

print(sum);
sum = sum +  ' ';


