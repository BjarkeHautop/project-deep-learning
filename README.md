# Project Deep Learning: Traffic Sign Recognition

**Main branch** is the stable version. Make changes in the **develop branch**. You can request a pull/merge to the main branch once significant improvements are made.

German traffic sign benchmark dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data) and clicking download. Upload the zip file to your google drive (I chose path German-traffic-signs/archive.zip) (so if you change location change corresponding line in python). 

Access the project notebook on Google Colab [here](https://colab.research.google.com/drive/13d3OxR17lEpdMIqDzQYMeGqTh-vXQh71?usp=sharing). This link is necessary as the repository is private.

## To-do list
- Maybe do data augmentation (Note: don't flip it! Signs can change meaning - small rotations are probably fine).
- Set up a bayesian CNN. For example Bayes by Backprop. See for example here for code example for Bayes by Backprop: https://github.com/PacktPublishing/Enhancing-Deep-Learning-with-Bayesian-Inference/tree/main/ch05/bbb
- Check robustness on data. Modify data (adverserial attack, out of distribtuion (OOD), ...)

## Contribution Guidelines
- Make changes in the **Google Colab**.
- If making experimental changes make your own copy (i.e. deleteting stuff to rewrite it). Once it is ready merge it together with our Google Colab when ready. 
- Add some text between your code.
