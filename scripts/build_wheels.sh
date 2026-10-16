#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

IMAGE_NAME=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc12-main

CONTAINER_NAME=cudaqx_wheel_builder
CONTAINER_STATUS=$(docker container inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null)

# Function to check if image was updated
check_image_update() {
    local pull_output
    
    echo "Checking for updates to $IMAGE_NAME..."
    
    pull_output=$(docker pull "$IMAGE_NAME" 2>&1)
    
    if echo "$pull_output" | grep -q "Image is up to date"; then
        echo "Image $IMAGE_NAME is already up to date"
        return 1
    elif echo "$pull_output" | grep -q "Downloaded newer image"; then
        echo "Image $IMAGE_NAME was updated"
        return 0
    else
        echo "Unable to determine if $IMAGE_NAME was updated"
        return 2
    fi
}

if check_image_update; then
  if [ "$CONTAINER_STATUS" = "running" ]; then
      docker stop $CONTAINER_NAME
      docker rm $CONTAINER_NAME
  elif [ "$CONTAINER_STATUS" != "" ]; then
      docker rm $CONTAINER_NAME
  fi
  CONTAINER_STATUS=""
fi

# Create the container if it doesn't exits.
if [ "$CONTAINER_STATUS" = "" ]; then
    docker run -d --name $CONTAINER_NAME $IMAGE_NAME tail -f /dev/null
    docker exec -it $CONTAINER_NAME /bin/sh -c "$(cat ./scripts/ci/build_cudaq_wheel.sh)"
fi

echo "Starting container..."
docker start $CONTAINER_NAME

echo "Copying CUDA-QX source to the container"
docker exec $CONTAINER_NAME rm -rf /cuda-qx
docker cp $(pwd) $CONTAINER_NAME:/cuda-qx

echo "Building CUDA-QX wheels in the container..."
docker exec -it $CONTAINER_NAME /bin/sh -c "$(cat ./scripts/ci/build_qec_wheel.sh)"

echo "Copying wheels from container..."
docker cp $CONTAINER_NAME:/wheels/ .

echo "Stopping container..."
docker stop $CONTAINER_NAME
