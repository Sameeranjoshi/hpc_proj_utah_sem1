#!/bin/bash

## declare an array variable
declare -a arr=("atb_par.c" "atb_par_unroll.c")

## now loop through the above array
for i in "${arr[@]}"
do
	filename=$(basename -- "$i")
	extension="${filename##*.}"
	filename="${filename%.*}"
   	gcc -O3 -fopenmp atb_main.c parallel_codes/$i -o exe/$filename.out
#	bash exe/$filename.out > reports/$filename.txt
 	inputfilename="input.txt"

	if [ -e reports/$filename.txt ]
           then
           	echo "Moving the previous file to reports/$filename.txt.old and please check if old data was useful else delete the old log file"
                mv reports/$filename.txt reports/$filename.txt.old
        fi

	echo "$filename" >> reports/$filename.txt
	 #Open the file
	 while read -r line; do
	     #process the line
#	     echo "Inputsizes: $line"
	     #Pass the line as input to another script
	     	    echo "$line" | ./exe/$filename.out  >> reports/$filename.txt
	 done< "$inputfilename"

done

