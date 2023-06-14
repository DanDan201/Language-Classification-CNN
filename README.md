# Language-Classification-CNN
#### This is the code for a language classification CNN (convolutional neural network). The data I'm working with is a collection of 5 seconds Youtube clips with 6 different languages: English, Spanish, Dutch, German, French, Portugese.
#### The dataset consists of individual waveforms. These contain the shape of the sound signal over time. Because our audio clips are digital, the waveforms consist of sound amplitude at invidual timesteps. The clips here are sampled at 8 kilohertz (khz), meaning there are 8000 amplitude measurements for every second. Each audio clip is 5 seconds long. Each audio clip has 8000 · 5 = 40000 measurements in total.
#### 
