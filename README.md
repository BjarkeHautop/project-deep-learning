# Project Deep Learning: Traffic Sign Recognition

German traffic sign benchmark dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data) and clicking download. Upload the zip file to your google drive (I chose path German-traffic-signs/archive.zip) (so if you change location change corresponding line in python). 

Access the project notebook on Google Colab [here](https://colab.research.google.com/drive/13sinqL_gKc4Pjr3Vthyoy9Oz-zUtiEKY?usp=sharing). This link is necessary as the repository is private.
The steps to build our CNN can be found in  this notebook: [here](https://colab.research.google.com/drive/1mkO0da_xT6EvOzx3tu7WttY9SrB04cR4?usp=sharing) 

Our project report is on Overleaf. It can be viewed and edited [here](https://www.overleaf.com/7537817147pnfzkmddskrb#ffc558).

## To-do list
- Maybe do data augmentation (Note: don't flip it! Signs can change meaning - small rotations are probably fine).
- Set up a bayesian CNN. For example Bayes by Backprop. See for example here for code example for Bayes by Backprop: https://github.com/PacktPublishing/Enhancing-Deep-Learning-with-Bayesian-Inference/tree/main/ch05/bbb
- Check robustness on data. Modify data (adverserial attack, out of distribtuion (OOD), ...)
- Write project report. 

## Considerations 
- Save trained models and load them. Training takes a long time already (and Bayesian CNN would take even longer). 

## Contribution Guidelines for Google Colab
- Make changes in the **Google Colab**. Once in a while download it as ipynb file and upload it here.
- If making experimental changes make your own copy (i.e. deleteting stuff to rewrite it). Once it is ready merge it together with our Google Colab when ready. 
- Add some text between your code.
