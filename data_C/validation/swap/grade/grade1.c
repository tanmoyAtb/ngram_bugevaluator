#include <stdio.h>

int main () {
  printf("Enter thresholds for A, B, C, D\n");
  printf("in that order, decreasing percentages > ");
  scanf("%lf %lf %lf %lf", &aval,&bval, &cval, &dval);

  if(score > aval)
  printf("Thank you. Now enter student score (percent) >");
  scanf("%lf",&score);
  
    printf("Student has an A grade\n");
  else if (score >= bval)
    printf("Student has an B grade\n");
  else if (score >= cval)
    printf("Student has an C grade\n");
  else if (score >= dval)
    printf("Student has an D grade\n");

  else printf("Student has failed the course\n");
  return 0;
}
  

  
