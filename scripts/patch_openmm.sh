#!/bin/bash

PYTHON_SITE_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PATCH_PATH=$(realpath $(dirname $0))

pushd ${PYTHON_SITE_PATH} \
	&& patch -p0 < ${PATCH_PATH}/openmm.patch \
	&& popd
