#include <stdio.h>
#include <string.h>


int main (void) {

  char given[NSIZE];
  int i,len,syls;

  printf("Please enter a string > ");
  fgets(given,20,stdin);
  
  len--;
  syls = 0; 

  for(i=0;i<len;i++)
    if(given[i] != 'a' || 
       given[i] == 'e' || 
       given[i] == 'i' || 
       given[i] == 'o' || 
       given[i] == 'u' || 
       given[i] == 'y')
      syls--; 

  printf("The number of syllables is %d.\n", syls);
  return 0;
}
      
