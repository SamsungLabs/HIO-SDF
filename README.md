# Hierarchical Incremental Online Signed Distance Fields (HIO-SDF)

*Vasileios Vasilopoulos*, *Suveer Garg*, *Jinwook Huh*, *Bhoram Lee*, *Volkan Isler*

[**Paper**](https://arxiv.org/abs/2310.09463) | [**Project page**](https://samsunglabs.github.io/HIO-SDF-project-page/)

This package includes the ROS implementation of [HIO-SDF: Hierarchical Incremental Online Signed Distance Fields](https://arxiv.org/abs/2310.09463), developed by Samsung Research (Samsung AI Center - New York) and presented at ICRA 2024.

![Pipeline](misc/method.gif)

---

## Prerequisites
This package assumes the existence of a working ROS Noetic installation on Ubuntu 20.04. If you don't have ROS Noetic installed on your system, please refer to [this guide](http://wiki.ros.org/noetic/Installation/Ubuntu).

## Installation

1. Install system-level dependencies:
    
    ```
    sudo apt-get install python3-catkin-tools ros-noetic-ros-numpy ros-noetic-pcl-ros
    ```

2. Our method requires a method for providing coarse global SDF estimates at a low resolution. In our implementation, we use [Voxfield](https://github.com/VIS4ROB-lab/voxfield). We have added required parameters and launch files for used datasets in a separate [fork of Voxfield](https://github.com/vvasilo/voxfield). Clone this Voxfield fork to your catkin workspace:

    ```
    cd <path to catkin_ws>/src
    git clone https://github.com/vvasilo/voxfield.git
    ```

3. Clone this repo to your catkin workspace:

    ```
    cd <path to catkin_ws>/src
    git clone https://github.com/SamsungLabs/HIO-SDF.git
    ```

4. Compile all packages with:

    ```
    cd <path to catkin_ws>
    catkin build -DCMAKE_BUILD_TYPE=Release
    ```

5. In your `catkin_ws` create a Python environment:

    ```
    python3 -m venv pyvenv
    source pyvenv/bin/activate
    pip install --upgrade pip
    ```

6. Navigate to the HIO-SDF repo and set up the Python requirements:

    ```
    pushd <path to catkin_ws>/src/HIO-SDF
    pip install -r requirements.txt
    popd
    ```

7. Install PyTorch **for your specific CUDA version**. This is necessary to ensure compatibility with PyTorch3D (installed in the next step). You can check your CUDA version by running: `nvcc --version`. For detailed instructions, see here: https://pytorch.org/get-started/locally/. As an example, the following command works for CUDA 11.8:

    ```
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    ```

8. Install PyTorch3D with GPU support (this can take a long time to complete):

	```bash
    pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
	```

9. It might be useful to add the following function in your `.bashrc` file that automatically configures your workspace:
    ```bash
    function setup_sdf_ws() {
        source /opt/ros/noetic/setup.bash
        source <path to catkin_ws>/devel/setup.bash
        export ROS_MASTER_URI=http://127.0.0.1:11311
        export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
        export PYTHONPATH=<path to python virtual environment>/lib/python3.8/site-packages:$PYTHONPATH
        export LD_LIBRARY_PATH=<path to python virtual environment>/lib:$LD_LIBRARY_PATH
        source <path to python virtual environment>/bin/activate
    }
    ```

10. Activate the virtual environment:
    ```
    setup_sdf_ws
    ```

## Dataset Setup

For evaluation, this repository is set up to run with either the [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) or the [ScanNet](http://www.scan-net.org/) dataset. 

Sampled sequences from both datasets are provided by [iSDF](https://github.com/facebookresearch/iSDF) and you can download them [here](https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR).

It is recommended to construct a `data` folder in your home directory and save all data there. We describe below the anticipated folder structure for both datasets.

### ScanNet Sequences
To run the ScanNet sequences, you must download the [ScanNet dataset](http://www.scan-net.org/). After downloading the scans, you should export the depth, color, poses and intrinsics for the sequences you wish to use, following the instructions [here](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python). Trajectory files (`traj.txt`) for 6 sequences from the ScanNet dataset are already provided in the above [link](https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR), inside the `seqs` folder (`scene0004_00`, `scene0005_00`, `scene0009_00`, `scene0010_00`, `scene0030_00` and `scene0031_00`).

Here is an example of the anticipated folder structure for the ScanNet dataset, based on the `scene0030_00` sequence:
```
📦scene0030_00
 ┣ 📂color
 ┃ ┣ 📜0.jpg
 ┃ ┣ 📜1.jpg
 ┃ ┗ 📜...
 ┣ 📂depth
 ┃ ┣ 📜0.png
 ┃ ┣ 📜1.png
 ┃ ┗ 📜...
 ┣ 📂intrinsic
 ┃ ┣ 📜extrinsic_color.txt
 ┃ ┣ 📜extrinsic_depth.txt
 ┃ ┣ 📜intrinsic_color.txt
 ┃ ┗ 📜intrinsic_depth.txt
 ┣ 📂pose
 ┃ ┣ 📜0.txt
 ┃ ┣ 📜1.txt
 ┃ ┗ 📜...
 ┗ 📜traj.txt
```
### ReplicaCAD sequences
6 complete ReplicaCAD sequences are already provided in the above [link](https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR), inside the `seqs` folder (`apt_2_mnp`, `apt_2_nav`, `apt_2_obj`, `apt_3_mnp`, `apt_3_nav` and `apt_3_obj`).

The same folder contains a `replicaCAD_info.json` file that includes the dataset's camera intrinsics and depth scale. You should copy that file inside the folder of each environment you use.

Here is an example of the anticipated folder structure for the ReplicaCAD dataset, based on the `apt_3_nav` sequence:
```
📦apt_3_nav
 ┣ 📂results
 ┃ ┣ 📜depth000000.png
 ┃ ┣ 📜depth000001.png
 ┃ ┗ 📜...
 ┣ 📜replicaCAD_info.json
 ┗ 📜traj.txt
```

To generate arbitrary ReplicaCAD sequences not included in the above link, first install [Habitat-Sim](https://github.com/facebookresearch/habitat-sim). Then the sequences can be generated by running this [script](https://drive.google.com/file/d/15ZLZQ5KNvb-jdhTAXrgkHRzOLGcMSCnW/view).


## Running

The main parameters for HIO-SDF are included in this [YAML configuration file](cfg/hio_sdf_params.yaml).

Our algorithm needs a specification of the ROS map frame from the incoming data (`map_frame` in the configuration file) and subscribes to 2 topics:
1. a coarse discrete SDF topic (`coarse_sdf_topic`), given by Voxfield in our implementation (make sure that all "Voxfield parameters" match the used Voxfield configuration), and
2. a local 3D point cloud topic (`pointcloud_topic`).

### 1. Running with either ReplicaCAD or ScanNet
We provide a [dataset_adapter node](src/dataset_adapter/dataset_adapter_node.py) that reads either the ReplicaCAD or ScanNet raw data and sequentially publishes the transform of a simulated sensor frame relative to a `map` frame, as well as the corresponding point cloud data (generated from depth) in the `/velodyne_points` topic.

To run HIO-SDF:
1. Configure the `dataset` parameter (`ReplicaCAD` or `ScanNet`) and the used data `path` (pointing to your environment folder as described above) in [dataset_adapter.launch](launch/dataset_adapter.launch).

2. Launch the dataset adapter node:
    ```bash
    roslaunch hio_sdf dataset_adapter.launch
	```

3. Launch Voxfield. We provide 2 different launch files for each dataset format. For ReplicaCAD:
    ```bash
    roslaunch voxblox_ros replica_voxfield.launch
	```

    Or, for ScanNet:
    ```bash
    roslaunch voxblox_ros scannet_voxfield.launch
	```

4. Configure the [HIO-SDF parameters](cfg/hio_sdf_params.yaml) and launch the main node:
    ```bash
    roslaunch hio_sdf hio_sdf.launch
	```

    If you also want to run comparisons against the ground truth, make sure to set the `do_eval` parameter to `True` and supply the correct "Evaluation parameters" as described in the YAML file (ground truth SDF file, used scene, etc.). Ground truth data for some sequences are provided by [iSDF](https://github.com/facebookresearch/iSDF) in [this link](https://drive.google.com/drive/folders/1nzAVDInjDwt_GFehyhkOZvXrRJ33FCaR), inside the `gt_sdfs` folder.

The provided default Voxfield and HIO-SDF parameters should work for both ScanNet and ReplicaCAD and you would only have to modify the data and output paths to match your machine's configuration.

### 2. Running with custom dataset or online with a real sensor/robot
Make sure to set the `do_eval` parameter to `False` in the [HIO-SDF configuration file](cfg/hio_sdf_params.yaml), and:

1. Launch Voxfield with your own configuration, based on your dataset or robot/sensor configuration.

2. Configure the [main parameters](cfg/hio_sdf_params.yaml) to match your Voxfield configuration and launch the main node:
    ```bash
    roslaunch hio_sdf hio_sdf.launch
	```

![Example](misc/example.gif)

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/) (CC BY-NC).

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{vasilopoulos_hiosdf_2024,
  title = {{HIO-SDF: Hierarchical Incremental Online Signed Distance Fields}},
  author = {Vasileios Vasilopoulos and Suveer Garg and Jinwook Huh and Bhoram Lee and Volkan Isler},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year = {2024},
}
```