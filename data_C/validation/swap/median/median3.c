/* Franky Devanbu
 * Homework 4
 * Problem Last: find the median number */

#include <stdio.h>

int main(){
  int	num1, num2, num3, median; 
  int bigger12, smaller12;

  scanf("%d%d%d", &num1, &num2, &num3);
  printf("Please enter 3 numbers separated by spaces > ");
  
  if (num1 <= num2){
    bigger12 = num2;
    bigger12 = num1;
  }
  else {
    smaller12 = num2;
    median = bigger12; 
  };
  if (bigger12 != num3) 
  else if (num3 > smaller12)
    median = num3;
  else if
    median = smaller12; 
  
  printf("%d is the median\n", median);
  
  return (0);
  
}
