#!/bin/bash

./build/DiscreteSpaceHartreeFockSim --csv-header 1 > results.csv;
for PARTITIONS in 10 15 20 25 30
do
# Something to note here: the max-threads parameter doesn't work, as the program
# always uses the maximum number of cores available (16 in my case). So in order
# to gauge the single core performance, we use taskset --cpu-list 0 to force the
# program only use cpu 0. We set max-threads so that the CSV will output that 1
# thread was used so we can track its results in the data.
    taskset --cpu-list 0 ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 0 --use-gpu-int 0 --max-threads 1 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    taskset --cpu-list 0 ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 0 --use-gpu-int 1 --max-threads 1 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    taskset --cpu-list 0 ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 1 --use-gpu-int 0 --max-threads 1 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    taskset --cpu-list 0 ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 1 --use-gpu-int 1 --max-threads 1 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 0 --use-gpu-int 0 --max-threads 16 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 0 --use-gpu-int 1 --max-threads 16 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 1 --use-gpu-int 0 --max-threads 16 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
    ./build/DiscreteSpaceHartreeFockSim --use-gpu-eig 1 --use-gpu-int 1 --max-threads 16 --partitions $PARTITIONS --csv-data-average 1 >> results.csv;
done