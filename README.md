# Detection of Stop Signs

## Abstract

In this project, we examine methods for detecting the presence of stop signs in photographs including:

-  Transfer Model with Google’s Inception V3
-  Mini-Inception Block Neural Network
-  Convolutional Neural Network
-  Logistic Regression using HOG

As stated in class, tackling the case of machine learning in autonomous cars in ten weeks would be infeasible, and so we opted for a related but more simple goal. On a broader spectrum, we hoped to achieve a deeper understanding of machine learning for self-driving cars. At the end of our complete analysis, we found that Mini-Inception Block Neural Network achieved the best accuracy at about 89%, 3% higher the fully retrained Inception V3 model. Retrained Inception managed an accuracy of 86% but was not robust to red non-stop-sign objects. Logistic Regression achieved 73% accuracy by identifying roads instead of stop signs, while Convolutional Feed Forward Neural Network were the worst performing at 71%.

