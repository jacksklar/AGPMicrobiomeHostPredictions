#!/bin/sh

path='/data/sklarjg/AmericanGutProj/FASTQ/HiSeq25005' 
out_path='/data/sklarjg/AmericanGutProj/FASTQ/HiSeq2500_temp'
for filename in $path/*.fastq.gz; do
	echo "$filename"
	base=`basename -s .fastq.gz "$filename"`
	echo "$base"
	echo "$out_path/$base.fastq"
	gunzip -c $filename  > $out_path/$base.fastq
	head -400000 $out_path/$base.fastq > $out_path/temp.txt
	mv $out_path/temp.txt $out_path/$base.fastq
	gzip $out_path/$base.fastq
done
