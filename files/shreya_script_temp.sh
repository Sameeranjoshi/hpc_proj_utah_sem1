#!/bin/bash
gcc -O3 -fopenmp atbt_main.c $1_par.c -o $1
#Set the filename variable
filename="delete.txt"
#Open the file
while read -r line; do
    #process the line
    echo "Processing line: $line"
    #Pass the line as input to another script
    echo "$line" | ./$1
done< "$filename"

