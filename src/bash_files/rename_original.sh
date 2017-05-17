#!/bin/bash
#Usage: bash rename_original.sh mask (being mask the folder in which files are stored and the name appended to the files)
cd $1
for f in *; do 
mv  "$f" "${f:0:6}_$1.nii.gz"
done
