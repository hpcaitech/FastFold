CUDA_VERSIONS=(11.2 11.4)
echo $CUDA_VERSIONS
IFS=','
DOCKER_IMAGE=()
for cv in ${CUDA_VERSIONS[@]}
do
DOCKER_IMAGE+=("\"hpcaitech/cuda-conda:${cv}\"")
done
echo ${DOCKER_IMAGE[*]}
container=$( IFS=',' ; echo "${DOCKER_IMAGE[*]}" )
container="[${container}]"
echo "$container"
echo "::set-output name=matrix::{\"container\":$(echo "$container")}"
