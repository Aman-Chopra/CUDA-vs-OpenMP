#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[])
{
	// Error check for the number of arguments
    if (argc != 2)
    {
        printf("usage: ./gen_examples n\n");
        printf("n = The number of numbers which needs to be sorted\n");
        exit(1);
    }

    // Taking the input from the command line
    int n = (int)atoi(argv[1]);

	FILE *fp_out;
	char name[50];

	// Creating the file
	sprintf(name, "%d", n);
	strcat(name, "input.txt");

	// Opening the file
	fp_out = fopen(name, "w");

	// The first line of the file is the number of elements to be sorted
	fprintf(fp_out, "%d\n", n);

	register int i = 0;

	// Writing the numbers to the file
	while(i < n)
	{
		int random = rand() % n;
		fprintf(fp_out, "%d ", random);
		i++;
	}

	return 0;
}
