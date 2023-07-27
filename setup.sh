#!/bin/bash

set -e

BASE=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $BASE

which python3
which pip3

for pkg in numpy matplotlib taskrun tol-colors; do
    if ! pip3 list 2>&1 | grep ${pkg}; then
	echo "pip installing ${pkg}"
	pip3 install ${pkg}
    else
	echo "${pkg} already installed :)"
    fi
done

if ! test -d calculon; then
    echo "Cloning calculon"
    git clone git@github.com:calculon-ai/calculon calc_proj
else
    echo "Calculon already cloned :)"
fi

COMMIT=84dea66
echo "Checking out commit ${COMMIT}"
cd calc_proj
git checkout ${COMMIT}
cd ${BASE}

echo "Environment successfully set up :)"
