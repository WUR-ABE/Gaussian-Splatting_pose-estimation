IMAGE_NAME=freshnet
IMAGE_TAG=maintainer
PYTHON_VERSION=3.7
CONDA_ENV_NAME=freshnet
UID:=$(shell id -u)
USER_NAME:=$(shell whoami)

docker-make:
	@docker build -t ${IMAGE_NAME}:${IMAGE_TAG} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
		--build-arg UID=${UID} \
		--build-arg USER_NAME=${USER_NAME} \
		-f ./Dockerfile \
		.