# RFBE

Bidirectional Enhancement and Robust Fusion for 3D Object Detection under Complex Lighting Conditions (The Visual Computer).

The code is based on [EPNet++](https://github.com/happinesslz/EPNetV2) and [PointRCNN](https://github.com/sshaoshuai/PointRCNN).

## Abstract

In diverse environmental conditions, the accuracy of vision-based 3D object detection is significantly impacted by varying illumination. Current multi-modal fusion methods combining cameras and LiDAR often suffer from sensor noise in raw data or overreliance on individual modalities during feature fusion, particularly in low-light settings. To address these challenges, we propose a robust fusion and bidirectional feature enhancement framework named RFBE. This framework achieves robust feature alignment and mapping through the integration of K-nearest neighbor (KNN) and cross-attention mechanisms, enabling comprehensive interactions between point cloud features and their corresponding image counterparts. Additionally, we incorporate a spatial encoder to generate attention gates for both point and image features, ensuring stable cross-modal feature generation and effective noise suppression. We further introduce an adaptive multimodal consistency loss to enhance detection accuracy under complex lighting conditions. Here we show competitive performance on the KITTI dataset and robust results in both custom low-light datasets and real-world low-light environments.

![image](image/KAM.png)

## Real-world low-light dataset
In order to verify the stability of the method, we built a multi-sensor acquisition device and measured the illuminance value in real time with the instrument, and completed the calibration work. Below is a partial working presentation of the dataset we produced.
![image](image/device.png)
![image](image/lux.png)
![image](image/data.png)

## Install(Same with [EPnet++])

The Environment：
* Linux (tested on Ubuntu 16.04)
* Python 3.7.6
* PyTorch 1.20 + CUDA-10.0/10.1

a. Clone the RFBE repository.
```shell
git clone https://github.com/happinesslz/EPNetV2.git
```

b. Create conda environment.

```shell
conda create -n epnet_plus_plus_open python==3.7.6
conda activate epnet_plus_plus_open
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and our provided [train mask](https://drive.google.com/file/d/1XqqErkAl8t72HM65PJfZC6SQ1oeBF4X0/view?usp=share_link) based on the [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset). Then organize the downloaded files as follows: 

```
RFBE
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes) & train_mask
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```

## Implementation
### Training & Inference
```shell
bash run_train_and_eval_epnet_plus_plus_car.sh
bash run_train_and_eval_epnet_plus_plus_ped.sh
bash run_train_and_eval_epnet_plus_plus_cyc.sh
```
## Acknowledgement
Thanks for the superior open-source project [EPNet++](https://github.com/happinesslz/EPNetV2) and [PointRCNN](https://github.com/sshaoshuai/PointRCNN). Thanks for all co-authors.

## Citation
If you find this work useful in your research, please consider cite:
The article "Bidirectional Enhancement and Robust Fusion for 3D Object Detection under Complex Lighting Conditions" is expected to be published in The Visual Computer.
```
@article{Tao2025RFBE,
  title={Bidirectional Enhancement and Robust Fusion for 3D Object Detection under Complex Lighting Conditions},
  author={Tao, Zhonghan and Zhou, Youjie and Wan, Yi and Liang, Xichang and Li,Yanan},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```
@article{liu2022epnet++,
  title={EPNet++: Cascade bi-directional fusion for multi-modal 3D object detection},
  author={Liu, Zhe and Huang, Tengteng and Li, Bingling and Chen, Xiwu and Wang, Xi and Bai, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
