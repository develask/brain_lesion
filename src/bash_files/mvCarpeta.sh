#!/bin/bash

cd $1
for f in *; do
mkdir "../${f:3:3}"
mv "$f" "../${f:3:3}/$f"
done
