#!/bin/bash

export DISBATCH_KVSSTCP_HOST=10.250.149.2:32895

if [[ $1 == '--mon' ]]
then
    exec /mnt/sw/nix/store/7kfpfhj104s5hynnhr5jzsrm25p4vwla-python-3.11.11-view/bin/python3 -m disbatch.dbMon /mnt/home/tparshakova/Documents/step-back/disbatch_logs/tasks_disBatch_260529140636_794
elif [[ $1 == '--engine' ]]
then
    exec /mnt/sw/nix/store/7kfpfhj104s5hynnhr5jzsrm25p4vwla-python-3.11.11-view/bin/python3 -m disbatch "$@"
else
    exec /mnt/sw/nix/store/7kfpfhj104s5hynnhr5jzsrm25p4vwla-python-3.11.11-view/bin/python3 -m disbatch --context /mnt/home/tparshakova/Documents/step-back/disbatch_logs/tasks_disBatch_260529140636_794_dbUtil.sh "$@" < /dev/null 1> /mnt/home/tparshakova/Documents/step-back/disbatch_logs/tasks_disBatch_260529140636_794_${BASHPID-$$}_context_launch.log
fi
