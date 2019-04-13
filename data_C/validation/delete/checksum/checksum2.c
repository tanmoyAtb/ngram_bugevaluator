#include <stdio.h>

int main () {

  char it;
  int status;



  while ((status != scanf("%c", &it)) != EOF && it != '\n') 

   sum = (sum + it) % 64;
  
  sum = sum + (long) ' ';

  printf("Check sum is %c\n", (char) sum);
  return 0;
}

