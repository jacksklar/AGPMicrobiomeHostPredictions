#!/bin/sh

filename='/data/sklarjg/AmericanGutProj/fastq_rems/fastq_remainder_'$1'.txt'
all_lines=`cat $filename`

for line in $all_lines;
do
        path="ftp://"$line
        echo $path
        wget $path
done
