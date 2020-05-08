#!/usr/bin/bash

# eliminate the ',,' problem (on all text files)
sed -i -- 's/,,/,/g' *.txt

# remove current header
tail -n +13 sift.txt > sift2.txt

# count number of columns
head sift2.txt -n 1 | grep -o "," | wc -w 

# generate new header
./add-att.py 90924 > new-header.txt

# concatenate new header with data
cat new-header.txt sift2.txt > sift.arff
