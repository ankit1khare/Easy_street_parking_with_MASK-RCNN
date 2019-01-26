# Easy_street_parking_with_MASK-RCNN
Pre-trained MaskRCNN from [Matterport](https://github.com/matterport/Mask_RCNN) has been used to detect cars in the parking. I've recorded a video of the parking near my apartment and even with my hands shaking due to cold, the overall prototype successfully detect an available parking space vacany. 

![](test_vid.gif)

Observe the change of color in the other parking spots. It is primarily due to moving camera while recording, the car parked in the area gets out of the marked spot. Using twilio's API, we can easy generate a number and use it to send a custom message to our own cell phone whenever there's a vacancy available to park. First, I assumed that the first frame will determine the parking spots and no car in the first frame would be a moving one. But then I identified the static cars by observing them for 5 seconds and assumed that they are parked in the parking area only.     
