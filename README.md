# AI-Powered History Teller iOS app

A mobile application that can detect the statue and play an embedded video within couple of seconds on the screen with no hassle, by just pointing the camera at it. This app makes it easy to learn the history behind the statues around University of Dayton without any extra effort.


Accuracy around 98% has been achieved for Training dataset and 94% has been achieved for Testing it in real-time. 


The dataset contains 6000+ images of 24 statues were manually collected with various shades of lighting, at different times and angles to make the model work with high accuracy in extreme conditions. 


The model is based on Transfer Learning. We have used EfficientNet Lite v4. It is accurate and is ideal to deploy on Mobile Platforms. We have trained the model and converted it to TFLite model, which we used in the final version of the app.


The app was initially developed on Unity 2019.4.32f1 additionally with ML and Barracuda packages. After the UI and assets have been developed, the Unity project is converted into iOS/Xcode project to deploy on iOS platforms. 


Video Demo - <https://youtu.be/I3Df4SEp_vE>


Presentation - <https://docs.google.com/presentation/d/1MyJuEaoocPWPz87hRQVK15RqANbiS22E/edit?usp=sharing&ouid=113556315390535528754&rtpof=true&sd=true>


Model Code - <https://github.com/vijaykvganaraju/statue-story-teller-app/blob/master/Model_Training_File/UDsculptures_Classification.ipynb>

