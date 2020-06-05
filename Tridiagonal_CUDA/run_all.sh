#!/bin/bash
#SBATCH -J Tridiagonal
#SBATCH -o out.txt
#SBATCH -p P100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 9:00:00

export KMP_AFFINITY=scatter

SUFFIX=`date +%D-%R`
MAXSIZE=134217728
REP=200

sizes=(64 128 192 256 320 384 448 512 1024 1536 2048 2560 3072 3584 4096 8192 12288 16384 20480 24576 28672 32768 65536)
algos=(0 1 3 4 9 14)

function run_executable() {
    CUR=`pwd`
    # Gorgon P100
    # srun -p P100 ./test $1 $2 $3 $4 $5 $6
    # 244 K40m
    # srun -p gpu ./test $1 $2 $3 $4 $5 $6
    # Lantern TITAN
    # srun -p gpu@cm.cluster ./test $1 $2 $3 $4 $5 $6
    # sbatch
    ./test $1 $2 $3 $4 $5 $6
}

RES=result_single_$SUFFIX.csv
for algo in ${algos[@]}
do
    for ii in ${sizes[@]}
    do
        tasks=$(($MAXSIZE/$ii))
        run_executable 0 $algo $ii $tasks $REP $RES
    done
done

algos=(0 1 3 4 9)

RES=result_double_$SUFFIX.csv
for algo in ${algos[@]}
do
    for ii in ${sizes[@]}
    do
        tasks=$(($MAXSIZE/$ii))
        run_executable 1 $algo $ii $tasks $REP $RES
    done
done
