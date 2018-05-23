# Robot Frame Registration
-----
## 1 Overview
this algorithm calculates the transformation of location of a surgical robot using the nurual network.

The robot has a frame to help the algorithm to quantify the transformation. According to the 2D images from MRI scan, the algorithm produces the 6 parameter of the transformation to get the new location of the robot's frame.

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

## 2 Project Structure
- networks
  > this archive contains nerual networks with different stucture
- play
  > this archive contains the tensorboard files while run the networks has been trained
- RawDataFullConnect
  >the rawdata
- save model 
  >this archive contains the parameters of a trained nerual network
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



