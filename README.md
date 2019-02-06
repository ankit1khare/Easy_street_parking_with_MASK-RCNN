# Easy_street_parking_with_MASK-RCNN
Pre-trained MaskRCNN from [Matterport](https://github.com/matterport/Mask_RCNN) has been used to detect cars in the parking. I've recorded a video of the parking near my apartment and even with my hands shaking due to cold, the overall prototype successfully detect an available parking space vacany. 

![](test_vid.gif)

Observe the change of color in the other parking spots. It is primarily due to moving camera while recording, the car parked in the area gets out from the marked spot. Using twilio's API, we can easy generate a number and use it to send a custom message to our own cell phone whenever there's a vacancy available to park. There's a great medium post [here](https://medium.com/@ageitgey/snagging-parking-spaces-with-mask-r-cnn-and-python-955f2231c400) which describes the process flow in a great depth. The underlying assumption is that, the first frame will determine the parking spots and no car in the first frame should be a moving one. 

![](assumption_test1.gif)



This is very inconvinient. We can't expect to take our cell phone out and get bluffed by a moving car just because it was in the first frame. So, I identified the static cars by observing them for 5 seconds and assumed that they are parked in the parking area only. This way, no moving cars would hamper out system. 

![](better_test1.gif)





The approach is pretty simple. I just took two frames and compared them for a possible motion using frame subtraction. Next I eroded the area occupied by the moving vehicle so that MASK-RCNN would not capture it. 




![](night_blur_test.gif)



Even in night with a bad camera quality it works just fine. How about a better camera?


![](night_antiblurc_test.gif)



Easy, right! Now all I have for you guys is to check my other Yolo repository to see how we can speed up the process using batch-processing. Essentially, I am asking you to read multiple frames, keep them in buffer and then send them for processing to GPU at once to maximze GPU utilization. This way you might be able to take advantage of colab's 12 GB of free K80. 

Have fun and do let me know if you come up with a modification.




