#!/bin/bash

clear_directories () {
  if [ -d "$RESULTS_DIR" ]; then
    rm -rf "$RESULTS_DIR"
    echo "Removing $RESULTS_DIR"
  else
    echo "$RESULTS_DIR does not exist"
  fi
  if [ -d "$LOGS_DIR" ]; then
    rm -rf "$LOGS_DIR"  
    echo "Removing $LOGS_DIR"
  else
    echo "$LOGS_DIR does not exist"
  fi
  if [ -d "$CKPTS_DIR" ]; then
    rm -rf "$CKPTS_DIR"
    echo "Removing $CKPTS_DIR"
  else
    echo "$CKPTS_DIR does not exist"
  fi
}

EXPERIMENT_NAME=$1

echo EXPERIMENT_NAME   : $EXPERIMENT_NAME

# BASE_DIR=$(pwd)/CL_NeRF
BASE_DIR=/workspace/CLNeRF/CLNeRF

RESULTS_DIR=$BASE_DIR/results/NGPGv2/CLNerf/colmap_ngpa_CLNerf/$EXPERIMENT_NAME
LOGS_DIR=$BASE_DIR/logs/NGPGv2_CL/colmap_ngpa_CLNerf/$EXPERIMENT_NAME
CKPTS_DIR=$BASE_DIR/ckpts/NGPGv2_CL/colmap_ngpa_CLNerf/$EXPERIMENT_NAME

echo RESULTS_DIR   : $RESULTS_DIR
echo LOGS_DIR      : $LOGS_DIR
echo CKPTS_DIR     : $CKPTS_DIR

echo "Are you sure you want to clear $EXPERIMENT_NAME?"
select yn in "YES" "NO"; do
    case $yn in
      YES ) clear_directories; break;;
      NO ) exit;;
    esac
done

