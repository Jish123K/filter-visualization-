# filter-visualization

File.py=

This code defines a function visualize_filter that uses gradient ascent to generate visualizations of the filters in a specified layer of a pre-trained VGG-16 model. The function takes an input image, the index of the filter to be visualized, and the name of the layer containing the filter as inputs. The function then applies gradient ascent to maximize the activation of the specified filter, while also applying several regularization techniques to prevent the generated image from being too noisy or too high-frequency. Finally, the function returns the generated image.

Utilt.py=

This code is implementing some utility functions for visualizing filters in a pre-trained VGG16 model. The main function is visualize_filter, which takes an input image, a filter index, a layer in the VGG16 model, and the number of iterations to run gradient ascent. It then performs gradient ascent on the input image to maximize the output of the filter at the specified layer. The save_filters function is used to save the generated filter visualizations to disk.
