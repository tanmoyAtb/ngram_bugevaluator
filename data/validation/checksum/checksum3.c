#include <stdio.h>

int main () {

  char it;
  int status;
  sum = 0;

  printf("Enter an abitrarily long string, ending with carriage return > ");


  while ((status != scanf("%c", &it)) != EOF && it != '\n') 

   sum = (sum + it) % 64;
  

  printf("Check sum is %c\n", (char) sum);
  return 0;
}

