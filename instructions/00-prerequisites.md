# Prerequisites

This document describes the libraries needed and how to be installed. 
Python 3 is required (tested with version 3.7).


## Installation Steps

1. Clone the current repository
    ```console
    user@ubuntu:~$ git clone https://github.com/athanell/segment-sah-from-ncct.git
    ```

2. Create and activate a virtual environment 
    ```console
    user@ubuntu:~$ conda create --name ncct_sah_seg python=3.7
    user@ubuntu:~$ conda activate ncct_sah_seg
    ```


3. Install dependencies
    ```console
    user@ubuntu:~$ pip install requirements.txt
    ```   

## Test the algorithm
Your installation is complete. Now you can [test](01-inference-unet.md) the algorithm with your ncct images.
