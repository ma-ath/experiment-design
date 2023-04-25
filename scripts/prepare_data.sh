#!/bin/bash
# mkdir dataset
cd dataset
# while read file; do
#     wget --no-check-certificate ${file} -b
# done < files.txt
# Extract zip files
unzip -o datasets/zip/\* -d datasets/extracted/