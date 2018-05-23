# Robot Frame Registration
## 1 Overview
this algorithm calculates the transformation of location of a surgical robot using the nurual network.

The robot has a frame to help the algorithm to quantify the transformation. According to the 2D images from MRI scan, the algorithm produces the 6 parameter of the transformation to get the new location of the robot's frame.

![](https://github.com/halimiqi/Frame_Registration/blob/master/images/frame.png)
![](https://github.com/halimiqi/Frame_Registration/blob/master/images/2Dimage.png)

> the initial location of frame


![](https://github.com/halimiqi/Frame_Registration/blob/master/images/frame2.png)
![](https://github.com/halimiqi/Frame_Registration/blob/master/images/2Dimage2.png)

> the location after transformation
## 2 Project Structure
- networks
  > this archive contains nerual networks with different stucture
- play
  > this archive contains the tensorboard files while run the networks has been trained
- RawDataFullConnect
  > the rawdata
- save model 
  > this archive contains the parameters of a trained nerual network
- tfboard
  > this archive contains tensorboard files while training
- config_full.py
  > the overall configuration
- GetMarkLocation.py
  > preprocess the rawdata
- Main_FullConnect.py
  > main function of the projects

## 3 Usage
### 3.1 To Train The Network
set the variable`self.IsTraining = True` in the `config_full.py`

run `Main_FullConnect.py`
### 3.2 To Run A Trained Network
set the variable `self.IsTraining = True` in the `config_full.py`

run`Main_FullConnect.py`

the output is`pred.npy`. It is an 2D numpy array, the first axis is the index of the images,the second axis is the transformation. The format of the transformation is `[translationX(mm), translationY(mm), translationZ(mm), rotationX°,rotationY°, rotationZ°]`





