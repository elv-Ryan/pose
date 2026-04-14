#!/bin/bash

buildscripts/build_container.bash -t "pose:${IMAGE_TAG:-latest}" -f Containerfile .