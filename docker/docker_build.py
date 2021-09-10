#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "docker build -t vainavi-keypoints ."
    code = os.system(cmd)
