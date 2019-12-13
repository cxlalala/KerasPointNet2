#!/usr/bin/env bash

# Use all GPUs, mount and work in CWD, run as current user, remove the container once finished with it.
container_cwd_mount_path="/cwd"
docker_default_args="
--gpus all
-v $PWD:$container_cwd_mount_path -w $container_cwd_mount_path
-u $(id -u):$(id -g)
--rm -it tensorflow/tensorflow:latest-gpu"

function train_net() {
    dataset_path="$1"
    trainer_args=${@:2:$#-1}
    docker run\
        -v $dataset_path:/data\
        $docker_default_args\
        python ./train.py /data/ $trainer_args
    }

function eval_net() {
    board_id="$1"
    data_mount_path="$2"
    out_mount_path="$3"
    model_dir="$4"
    docker run\
        -v $data_mount_path:/data -v $out_mount_path:/out\
        $docker_default_args\
        python ./eval.py\
        /data/$board_id.Geo-{Top,Bottom,Left,Right}.ltiDepthMapStream\
        /out\
        $model_dir
    }

function compile_net_ops() {
    docker run $docker_default_args bash -c "
    pushd custom_ops/
    make -j
    popd
    "
}
