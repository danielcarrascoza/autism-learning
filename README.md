# autism-learning
Lesson generator using live attention detection and student history, tailored for neuro-diverse learners. Buildathon: Loveable/Good Vibes Only

ROOM FOR IMPROVEMENT/PROBLEMS FOR ANOTHER DAY:

It won't deploy because Mediapipe is not supported, and it was not able to switch runtimes.

1. Use data of actual students to get a good background weight instead of generated fake weights.
2. Ditch this dataset and use another that has more movement. This dataset was very much the same all the time, with high engagement, making the attention perform worse
3. An extension of the last one, find actual neurodivergent K-12 students to better understand and manually write labels for a better dataset that can then be used to perform this.
4. Make lessons based on item 1 - generation

I believe that having an attention meter that is trained on actual students on the spectrum would be beneficial to teachers who struggle with teaching neurodivergent kids. 
This can be done by analyzing student data from a school, getting permission to film and label students' attention, and researching good lesson plans 

This repository has a neural network that grabs from features. The dataset is poor and did not capture enough feature mapping from it.
--> This could be improved by a better dataset, or, potentially, training a better model that doesn't just take in features and instead would run directly from camera capture.
