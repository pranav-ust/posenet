# Pose Net

Simplified version of ECCV paper (Simple Baselines for Human Pose Estimation and Tracking) which you can use that for your custom files! Extremely fast and easy to use.

## Requirements

Python 3.5+
Pytorch >= 0.40
OpenCV 4.0
CuDA

## Usage

First clone this repository and cd into it.

```
git clone https://github.com/pranav-ust/posenet.git
cd posenet
```

Then, using these commands install the models.

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dE-tcTGRriBeiEBvHRjEG1z9DJmN7CLS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dE-tcTGRriBeiEBvHRjEG1z9DJmN7CLS" -O pretrained.tar.gz && rm -rf /tmp/cookies.txt
tar -xvzf pretrained.tar.gz
```

Disable cudnn for batch_norm:

```
# PYTORCH=/path/to/pytorch
# for pytorch v0.4.0
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
# for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

Note that instructions like # PYTORCH=/path/to/pytorch indicate that you should pick a path where you'd like to have pytorch installed and then set an environment variable (PYTORCH in this case) accordingly.

Run the file `python3 estimate.py imagename`

More details are as follows:

```
usage: estimate.py [-h] [--output OUTPUT] [--threshold THRESHOLD]
                   [--thickness THICKNESS]
                   image

positional arguments:
  image                 the image that you want to input

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       the output filename
  --threshold THRESHOLD
                        probability of the keypoint that should appear greater
                        than this threshold (between 0.0 to 1.0)
  --thickness THICKNESS
                        thickness of the line (from 1.0 to 10.0)
```

If the input is `obama.jpg`:

<img src="https://github.com/pranav-ust/posenet/blob/master/obama.jpg" alt="obama" width="100" height="200">

Output is:

<img src="https://github.com/pranav-ust/posenet/blob/master/output.jpg" alt="obama" width="100" height="200">

## Credits

Background code is based on [this repo on Simple Baselines for Human Pose Estimation and Tracking.](https://github.com/Microsoft/human-pose-estimation.pytorch)
