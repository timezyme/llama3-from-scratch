#!/bin/bash
set -e

cd "$(dirname "$0")/.."

make tests

for i in 1 2 3 4 5 6 7; do
    ./bin/tests "$i"
done
