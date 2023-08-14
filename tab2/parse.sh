#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Bad number of arguments"
    return -1
fi

echo "Input: ${1}"
echo "Output: ${2}"

cat ${1} | head -n 35 | tail -n 8 | column -t -s, > ${2}

