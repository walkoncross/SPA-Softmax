#!/usr/bin/env sh
GPU_ID=$1

./../tools/caffe-sphereface/build/tools/caffe train \
-solver /disk2/zhaoyafei/sphereface-train-merge_webface_asian/sphereface-20/sphereface_20_train_val_solver.prototxt \
-gpu ${GPU_ID} 2>&1 | tee -a result/train-log.txt

#-snapshot ./result/senet_sphereface_model_iter_64000.solverstate