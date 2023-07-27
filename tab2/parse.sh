#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Bad number of arguments"
    return -1
fi

cat ${1} | tail -n 9 | column -t -s, > ${2}

