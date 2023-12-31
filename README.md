# Object Detector

This is a program that allows you to get detections both on images from a folder and real-time broadcasts.
It works with any kind of objects which you can easily label and feed to the model.  

But to make the program work you have to follow a series of steps.  

## Download and prepare data

If you download data from this repository, you'll already have data from the **TensorFlow Model Garden** [official repository](https://github.com/tensorflow/models)
and also **[labelImg](https://github.com/heartexlabs/labelImg)** to easily label your images.  

But there is also some additional data you have to download and prepare:  

1. Follow the link to the **Detection Model Zoo** [official repo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
Select the **SSD ResNet50 V1 FPN 640x640 (RetinaNet50)** model and download it. This is a .tar archive so you'll need to untar it a little bit later.

2. Untar (decompress) the file you've just downloaded. Save the checkpoint folder to the test_data folder **_models/research/object_detection/test_data/_**

3. Download [this](https://github.com/protocolbuffers/protobuf/releases/download/v3.12.4/protoc-3.12.4-win64.zip) protoc file.
Extract the contents of the zip file. You'll see that bin directory contains the protoc.exe. Copy this file and save it to your Python\Scripts folder.
Then re-open the cmd, activate your venv (!), navigate to your project folder and run
`cd models/research && _protoc object_detection/protos/*.proto --python_out=.`
to compile the Object Detection API protocol buffers.

5. **Make sure that your venv is activated (!!).** Then run the [setup script](https://github.com/tensorflow/models/blob/master/research/object_detection/packages/tf2/setup.py) from the tensorflow models repo with `python -m pip install models/research.` If everything is OK, it'll take some time to setup all the required libraries. Wait a bit.
