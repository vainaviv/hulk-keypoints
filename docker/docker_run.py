#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = 'nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH -v %s:/host \
                                 -v %s:/host/data vainavi-keypoints' % (os.path.join(os.getcwd(), '..'), '/raid/vainavi/data/long_cable')
    #cmd = "nvidia-docker run -it priya-keypoints" 
    #cmd = "docker run --runtime=nvidia -it -v %s:/host priya-keypoints" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
