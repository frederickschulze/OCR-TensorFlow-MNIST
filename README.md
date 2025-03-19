In this notebook, I explore using TensorFlow to recognize handwritten digits from the MNIST dataset. 
To see the full code, please look at the notebook, but I will go into some of the explanation here.
The MNIST dataset is a set of 60,000 black and white 28x28 pixel images of handwritten digits. 
Creating an AI model that can recognize these digits is often considred the "Hello World" of deep learning.
To accomplish this goal, I'm going to use a convolutional neural network model, which is a popular approach for 
image recognition due to the efficiency in pattern recognition that convolution provides. 

Here is the model, the heart of the program: 

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )

The model is successfully trained with an accuracy of 98.75%. Here are some predictions on 10 random numbers made by the trained Neural Network:

<p align="center">
    <img src="https://github.com/user-attachments/assets/e03972d5-36c7-42ea-a40e-bef118f21de3" style=width:75%>
</p>

The is an interesting example becuase it actually shows a number that was incorrectly classified. The 6 does look vaguely similar to an 8. 
However, with further training and tweaking, it should be able to get a number like this correct. 

I wanted to see how the model did with different types of numbers, so I tallied up the predictions vs the actual images. While the examples are randomly chosen without replacement, 
I wanted to make sure to have 800 examples of each actual number. Seaborn has a great plot for this type of comparison called a confusion matrix (aptly named).
<p align="center">
    <img src="https://github.com/user-attachments/assets/3b64d0d2-cc62-4eeb-9a39-5605d91fb09a" style=width:75%>
</p>

The problem here is that the correct results dominate the plot. So here is another plot with the correct results removed:
<p align="center">
    <img src="https://github.com/user-attachments/assets/0213ac37-9710-4847-998d-2df22b0d839d" style=width:75%>
</p>

The largest value in this plot is the number 10: 2 was incorrectly identified as a 7 in 10 cases. Since there are 800 images of the number 2, this means that the model misclassifies a number 2 as a 7 in 10/800 cases, or 1.25% of the time. I thought it would be interesting to look at the cases where the model failed. Funnily enough, many of these images do look somewhat like the number 7.   
Here are those 10 cases:
<p align="center">
    <img src="https://github.com/user-attachments/assets/bde05155-11b4-4dd0-83e4-7cc99a747851" style=width:75%>
</p>

All in all, the model's accuracy is fairly good. I've summarized the accuracy at detecting each number in the below table. 
<p align="center">
    <img src="https://github.com/user-attachments/assets/b37e1f47-8920-4d4a-9bca-0889835b4874" style=width:40%>
</p>

Even the worst numbers, 2 and 9, can still be correctly identified by the model over 97% of the time. 
Since 99% is an industry benchmark accuracy for the MNIST dataset, I'd say this model is successful, especially since this model can be trained in only a minute or so on Google Colab. The convolution layers definitely help this accuracy. They enhance the pattern recognition 
of the model and aren't computationally expensive. This allows the "model" to only be 10 neurons, which is the key for it being able to be trained so quickly. After training, the lightweight model also has benefits in the use-phase since it uses minimal computation when actually identifying these digits as well. 

For better accuracy, I may try spending a longer time training the model locally. 
With more compute time and power, a more complex model can be used, which could improve these results even more. A lot more than 10 neurons 
can be used in order to enhance accuracy. 
