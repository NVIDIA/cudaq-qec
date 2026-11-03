#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2022 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

IMAGE_NAME=ubuntu:22.04

# Helper to stop and remove the container
cleanup() {
    echo "Stopping and removing container..."
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
}

docker pull $IMAGE_NAME
if [ $? -ne 0 ]; then
    echo "Failed to pull image $IMAGE_NAME"
    exit 1
fi

echo "Creating and starting temporary container..."
CONTAINER_ID=$(docker run -d $IMAGE_NAME tail -f /dev/null)
if [ $? -ne 0 ]; then
    echo "Failed to create and start container"
    exit 1
fi

echo "Copying workspace into the container..."
docker cp $(pwd) $CONTAINER_ID:/cuda-qx
if [ $? -ne 0 ]; then
    echo "Failed to copy source"
    cleanup
    exit 1
fi

echo "Testing wheels in the container..."
docker exec -it $CONTAINER_ID /bin/sh -c "$(cat ./scripts/ci/test_wheels.sh)"

if [ $? -ne 0 ]; then
    echo "Something went wrong."
    # If we are in a terminal, then we can start a interactive shell
    if [ -t 1 ]; then
        echo "Starting interactive shell.."
        docker exec -it $CONTAINER_ID /bin/bash
    fi
fi

cleanup
