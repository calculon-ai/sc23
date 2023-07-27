#!/bin/bash

set -e

BASE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $BASE

which python3
which pip3

for pkg in matplotlib numpy pandas taskrun tol-colors; do
    if ! pip3 list 2>&1 | grep ${pkg}; then
	echo "pip installing ${pkg}"
	pip3 install ${pkg}
    else
	echo "${pkg} already installed :)"
    fi
done

CALC_DIR=calc_proj
if ! test -d ${CALC_DIR}; then
    echo "Cloning calculon"
    git clone git@github.com:calculon-ai/calculon ${CALC_DIR}
else
    echo "Calculon already cloned :)"
fi

COMMIT=84dea66
echo "Checking out commit ${COMMIT}"
cd ${CALC_DIR}
git checkout ${COMMIT}
cd ${BASE}

echo "Environment successfully set up :)"
