Generate saliency maps for model classifications of [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) images.

Uses the RISE technique outlined [here](https://github.com/eclique/RISE).


### Reading List

Title | Author | Conf | Notes | Link
----- | ------ | ---- | ----- | ----
|  **RISE: Randomized Input Sampling for Explanation of Black-box Models.** | V Petsiuk, A Das, K Saenko  | BMVC 2018 | RISE: Saliency Technique for Blackbox models | http://arxiv.org/abs/1806.07421
| **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** | RR Selvaraju, M Cogswell, A Das, R Vedantam, D Parikh, D Batr | ICCV 2017 | GRAD-CAM technique for saliency which tracks gradient changes by sampling feature maps. | https://arxiv.org/pdf/1610.02391.pdf |
| **On Guiding Visual Attention with Language Specification** | S Petryk, L Dunlap, K Nasseri, J Gonzalez, T Darrell, A Rohrbach | arXiv preprint arXiv:2202.08926 | Training for Saliency with augmented loss functions. | https://arxiv.org/pdf/2202.08926.pdf |
| **"Why Should I Trust You?": Explaining the Predictions of Any Classifier** | Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin | KDD2016  | Learns an interpretable model locally around the prediction | https://arxiv.org/pdf/1602.04938.pdf

Run [setup.sh](https://github.com/dwil2444/DNN_Attention/blob/master/setup.sh) to install the dependencies required.

The [train.py](https://github.com/dwil2444/DNN_Attention/blob/master/train.py) script will 
train a CNN for classification on Fashion-MNIST. 

The [rise_mnist.py](https://github.com/dwil2444/DNN_Attention/blob/master/rise_mnist.py) script
will produce a sample saliency map given a single example from the Fashion-MNIST
validation set. (Everything is included in the [rise_mnist.ipynb](https://github.com/dwil2444/DNN_Attention/blob/master/rise_mnist.ipynb) notebook.