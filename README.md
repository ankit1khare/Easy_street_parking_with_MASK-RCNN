# Easy_street_parking_with_MASK-RCNN

A pre-trained Mask-RCNN model taken from Matterport's repo can easily be used to detect cars in a parking lot. To test this, I recorded a video of the parking near my apartment. Despite my hands shaking from the cold, the prototype was able to successfully detect an available parking space vacancy.

<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/test_vid.gif?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
      <em>I apologize for the shaking in the video, it was really cold outside</em>
    </center>
</p>
<br>

Notice how the color of some of the other parking spots changes. This is due to the movement of the camera while recording, the car that was parked in that area is taken out from the marked spot. Using the Twilio API, we can easily generate a number and use it to send a custom message to our own cell phone whenever there is an available parking space. There is a great Medium post that describes the process flow. The assumption is that the first frame will determine the parking spots and that there should not be any moving cars in the first frame.

<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/assumption_test1.gif?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
      <em>Assumption: The first frame will determine the parking spots and no car in the first frame should be in motion</em>
    </center>
</p>
<br>


This assumption is inconvenient. We cannot expect to take our cell phone out and be deceived by a moving car just because it was in the first frame. Therefore, we need to come up with a better solution. What if we identify static cars by observing them for 5 seconds and assuming that they are parked in the authorized parking area only? This way, no moving cars will disrupt our system.

<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/better_test1.gif?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
      <em>Observe the passing car at the beginning of the video. Our new method is working great!</em>
    </center>
</p>
<br>


```python
    while video_capture.isOpened():
        success, frame = video_capture.read()

        if not success:
            print("couldn't read video")
            break

        elif counter<40:
          #create another video reader object to compare the two frames   and verify the possibility of motion
          success, frame2 = video_capture.read()
          d = cv2.absdiff(frame, frame2)  
          grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
          blur = cv2.GaussianBlur(grey, (1, 1), 0)
          ret, th = cv2.threshold( blur, 20, 255, cv2.THRESH_BINARY)

          #perform these morphological transformations to erode the car which is moving so that it is not detected by MASKRCNN. Take the erosion levels to be high. 
          dilated = cv2.dilate(th, np.ones((30, 30), np.uint8), iterations=1 )
          eroded = cv2.erode(dilated, np.ones((30, 30), np.uint8), iterations=1 )

          #fill the contours for even a better morphing of the vehicle
          img, c, h = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          frame2 = cv2.drawContours(frame2, c, -1, (0,0,0), cv2.FILLED)

```

The approach is pretty simple. I took two frames and compared them for a possible motion using Adaptive Background Learning. Next I eroded the area occupied by the moving vehicle so that MASK-RCNN would not capture it.
<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/1_x6wTWuWlwlnic30Mj61S0g.png?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
      <em>This frame makes the operations performed in the above code very intuitive I think. For full code, check park_clever.ipynb</em>
    </center>
</p>
<br>



Let's see how well our system performs at night, just for fun :)

<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/night_blur_test.gif?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
      <em>Thanks to Mask RCNN, it works well even at night with a low quality input video.</em>
    </center>
</p>
<br>

What if we use iPhone 7 plus ? let's see:
<p>
    <img src="https://github.com/ankit1khare/ankit1khare.github.io/blob/master/_posts/gifs/night_better_test.gif?raw=true" style="max-width:100%;display: block;margin-left: auto;margin-right: auto;" alt>
    <center>
    <em>It's far better! It's amusing how the leftmost car is identified by MASK-RCNN with full confidence as soon as the headlights of the Camry focus on it.</em>
    </center>
</p>
<br>

Easy, right! Now, all I have for you is to check my other Yolo repository to see how we can speed up the process using batch processing. Essentially, I am asking you to read multiple frames, keep them in a buffer, and then send them for processing to the GPU at once to maximize GPU utilization. This way, you may be able to take advantage of Google Colab's 12 GB of free K80. Have fun and let me know if you come up with any further cool ideas! You can find the code on my Git. The code is runnable on Google Colab. Check out my [tutorial video on YouTube](https://www.youtube.com/watch?v=1aNT6S_VBNc) where I setup the colab and code the whole thing while explaining the logic behind it.

If we use something lighter like ThinYOLO (computationally efficient and good accuracy) and adopt a pub-sub type architecture to subscribe to such an application, it should work well in a real-time parking setting without much retraining and maintenance cost. Moreover, there might not be any permission/compliance issues as in case of installing sensors.

