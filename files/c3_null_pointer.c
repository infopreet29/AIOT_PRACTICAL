#include <stdio.h>
#include <stdlib.h>

int main() {
    int *ptr = NULL;
    *ptr = 42;   // dereferencing a null pointer
    printf("Value: %d\n", *ptr);
    return 0;
}
