IMAGE_NAME=gsplat_renderer
IMAGE_TAG=maintainer
PYTHON_VERSION=3.8
CONDA_ENV_NAME=gsplat_renderer
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