# Enhancing annotations for 5D apple pose estimation through Gaussian Splatting

## About
Official implementation of the paper "Enhancing annotations for 5D apple pose estimation through Gaussian Splatting"

Data is available at https://doi.org/10.4121/976c94f2-028f-4291-adfd-20eb82b0f647

## Installation

### Docker for rendering splat
1. Build Image

    ```
    cd docker_rendering
    make docker-make
    ```
2. Build the Container from the image

    Set the memory size depending on availability on your machine

    ```
    cd dvc-5d-apple-pose-estimation
    docker run --name gsplat_renderer --runtime=nvidia -it -v $PWD:$PWD -p 8080:8080 -m 20G --gpus all --shm-size=32G gsplat_renderer:maintainer /bin/bash
    ```

3. Install GSplat

    Next, GSplat needs to be installed within this container

    To install gsplat, run this command:
    ```
    pip install gsplat==1.5.0
    ```

    Perform the setup by calling rasterization with random data. 
    First, start a Python terminal with `MAX_JOBS=1`:
    ```
    export MAX_JOBS=1 && python
    ```

    Then, within this Python terminal, run the following to call the rasterization with random data:

    ```python
    import torch
    from gsplat.rendering import rasterization
    colors, alphas, meta = rasterization(torch.randn((100, 3), device="cuda"), torch.randn((100, 4), device="cuda"), torch.randn((100, 3), device="cuda") * 0.1, torch.randn((100, 3), device="cuda"), torch.randn((100,), device="cuda"), torch.eye(4, device="cuda")[None, :, :], torch.tensor([[300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device="cuda")[None, :, :], 300, 200)
    ```



### Updated docker for FRESHNet
1. Build image
     
    ```
    cd freshnet/docker
    make docker-make
    ```

2. Run Container

    Set the memory size depending on availability on your machine
     
    ```
    cd dvc-5d-apple-pose-estimation
    docker run --name freshnet --runtime=nvidia -it -v $PWD:$PWD -v /mnt/wur-w:/mnt/wur-w -p 8888:8888 -m 20G --gpus all --shm-size=32G freshnet:maintainer /bin/bash
    ```

3. Install the FRESHNet repository within the container.

   ```
   cd freshnet
   pip install -v -e .
   ```

4. Install MinkowskiEngine within the container
    
    First, dependencies need to be fixed for MinkowskiEngine

    ```
    sudo cp /usr/local/envs/freshnet/lib/libopenblas.so* /usr/local/envs/freshnet/lib/python3.7/site-packages/torch/lib/.
    ```

    Then, the engine can be installed:
    ```
    export MAX_JOBS=1 && pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
    ```

### Conda environment for other scripts
Some of the scripts are ran in a local Anaconda environment. To do this, install the environment from the `environment.yml` file using the following command:

```
conda env create -f src/local_environment/environment.yml
```

### Download dataset
The dataset can be downloaded from https://doi.org/10.4121/976c94f2-028f-4291-adfd-20eb82b0f647

The data can be placed in the folder `data_disk/dvc_data`, from which the contents can be excluded from GIT. 

## Usage
For this paper, [DVC](https://doc.dvc.org/) was used to build the pipeline and perform the experiments. In `dvc.yaml`, the stages of the pipeline are written, for using the rendered, original, or both datasets. 

The DVC project can be initialized using [`dvc init`](https://doc.dvc.org/command-reference/init). 

The `dvc.yaml` includes a specific folder structure, which was used to perform the experiments. Therefore, it is recommended to reuse the folder `data_disk/dvc_data`. The matched images

For the pipeline, the data in `02_matched_images`, `03_gaussian_splat\complete_gs`, `03_gaussian_splat\per_tree_gs`, and `04_annotations` need to be added to DVC using [`dvc add`](https://doc.dvc.org/command-reference/add)

The different stages can be ran using [`dvc repro`](https://doc.dvc.org/command-reference/repro). Parameters can be changed in `params.yaml`. 

## Citation
```
@article{
    
}
```
