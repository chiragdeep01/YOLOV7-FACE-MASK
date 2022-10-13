
# YOLOV7-FACE-MASK




## Installation

You can clone the repo by:

```bash
git clone https://github.com/chiragdeep01/YOLOV7-FACE-MASK.git
```
Then create an conda enivroment and install the requirements.txt file given in the repo by:
```cmd
pip install -r requirements.txt
```
Now your enviroment is ready along with the files.
    
## Usage

First you need to download the weights from the drive [Link](https://drive.google.com/file/d/1voiuoNuaSCfpxp7NYbXnCsZgP05qsQGd/view?usp=sharing).
You can also train you own weights but read the training warnings below.

Now open the config.json file and set weights to the path of the weights that you have downloaded.

Next you need to set up the cameras. In the json file cam1 is the name of the camera which you can give anything and 0 is for webcam.

```json
{

  "cameras" : {
              "cam1": 0
            },

  "weights" : "mask_weights/best.pt"
}

```
You can also setup your phone camera as an ip camera on your local network and use that to, for example:
```json
"cam1" : 0,
"cam2" : "http://192.168.1.18:8080/video"

```
cam2 is the phone camera that i have set up.

You can set up multiple cameras you will just have to add them to this json file.

Now open up the teminal and cd to the repo and activate your enviroment and then run:
```cmd
python main.py
```
Two directories will be created named feed and results. In the feed folder all the camera feed will come and in the results the detections will come sorted by camera name.


## Custom Training
If you plan to do a custom training make sure that you use only 2 classes nomask and mask in which the first class should be nomask and second should be mask.

The reason for this is that i have assigned colour to the detection based on class id so the name can be different but the number of classes should remain 2 and order should be the same.

If you have any doubt about custom training you can raise an issue and will help you out surely.

[YOLOv7](https://github.com/WongKinYiu/yolov7).
