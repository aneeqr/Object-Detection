# Talk Summary

1. We see that Deep Learning has played a major part in revolutionising Computer Vision. Computer vision techniques remain relevant to date and have improved Deep Learning. Which technique to use depends on your specific application. It is also true that Deep Learning is limited to data in some domains.

2. We also observed how Convolutional neural networks (CNNs) are instrumental for Object Detection and serve as feature extractors. To understand more on CNNs I encourage you to play with this really nice interactive visualisation  (https://poloclub.github.io/cnn-explainer/) to understand CNNs in more detail. Paper (https://arxiv.org/abs/2004.15004)

3. We also observed that CNNs which advanced state of the art on the ImageNet classification challenge, suffered from an inflection point, where the complexity costs, start to outweigh gains in accuracy. (https://arxiv.org/abs/1605.07678). Also, we discussed how most of the computation happens in these early convolution layers and the idea of producing these 2-D activation maps. Each point in these activation maps shows how your input image has reacted to your convolutional filters.

4. Following on from 3, we discussed that we need to have the appropriate hardware for Deep Learning and discussed the concept of GPUs and TPUs. We looked at different vendors, some of which were on-premises ones and the other options available on cloud.

5. In the on-premises one, we see how NVDIA offers the highest level of support and is a highly recommended platform if you are new to GPUs. Also, if you are planning on buying GPUs this year, have a look at the last Ampere architecture released by NVDIA. This database basically (https://www.techpowerup.com/gpu-specs/) will keep you posted on these releases. Also there is this recent very good analysis (https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/) by Tim Detters on his experience on selecting GPUs which I recommend you to read. In general, make sure you have a sufficent amount of Ram for appropriate batch size selection, Tensor Cores, a compute capability of greater than 7 and a high memory bandwidth.

6.  In the cloud options, we talked about this idea of spot-instances which are effective for prototyping of medium to small models In addition to this, we also talked about Colabs (https://colab.research.google.com/notebooks/intro.ipynb#recent=true) and it being a free service for quick prototyping. Among the cloud providers, Google Cloud Platform benefits from having both GPUs and TPUs and it allows you to connect your GPUs to their instances. Costs among these cloud providers vary depending on your usage!

7. In Deep Learning software frameworks, we talked about Pytorch and Tensorflow being the mainstream contenders. For beginners, Keras integrates seamlessly with Tensorflow and it is easy to understand and get your models running quickly.

8. We touched on Object Detection in general being a hard problem and there are a lot of choices. We have two object detection methods- Single shot object detection vs Two shot object detection. Single Shot Object Detectors are faster whereas Two shot detectors are more accurate.

9. We also touched upon the idea of transfer learning and talked about the idea of using pre-trained models to mitigate out computational costs. Fine tuning- which is one form of transfer learning benefits from using the fully connected layers for decision making while leveraging weights in the earlier layers. It is an easy to implement and a powerful technique.

10. We then quickly used existing frameworks and techniques to build an object detector using a few lines of code. OpenCV is a useful open source computer vision library and has supports for deep learning frameworks. While we cannot train our own models, we can definitely leverage support for pre-trained models.


# Object-Detection

For jupyter notebook use follow the following commands to set up your environment in windows using anaconda prompt. Check versions from requirements.txt.

1. conda create -n TF_object_detection2 pip python=3.6
2. activate TF_object_detection2
3. conda install notebook ipykernel
4. ipython kernel install --user --display-name tf_obj2
5. conda install -c conda-forge tensorflow
6. conda install -c ostrokach-forge googledrivedownloader 
7. pip install tqdm
8. pip install -U scikit-learn
9. pip install selenium
10. pip install pandas
11. pip install Pillow
12. pip install imutils
13. pip install matplotlib
14. pip install opencv-python
15. pip install seaborn
16. pip install urlib3
17. pip install numpy

Steps 5 to 17 with their respective versions should be:

5. conda install -c conda-forge tensorflow==1.14.0
6. conda install -c ostrokach-forge googledrivedownloader 
7. pip install tqdm==4.48.2
8. pip install -U scikit-learn==0.23.2
9. pip install selenium==3.141.0
10. pip install pandas==1.1.0
11. pip install Pillow==7.2.0
12. pip install imutils==0.5.3
13. pip install matplotlib==3.3.0
14. pip install opencv-python==4.3.0.36
15. pip install seaborn==0.10.1
16. pip install urlib3==1.25.10
17. pip install numpy==1.19.1

# Useful Resources

* **Deep Learning for Computer Vision by Justin Johson** https://web.eecs.umich.edu/~justincj/teaching/eecs498/
* **PyImageSearch by Adrian** https://www.pyimagesearch.com/
* **GPU Specs Database** https://www.techpowerup.com/gpu-specs/
* **Colab** https://colab.research.google.com/notebooks/intro.ipynb#recent=true
* **CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization** https://poloclub.github.io/cnn-explainer/
* Check out **Monk** which serves as a unified wrapper for TensorFlow, Keras, PyTorch or MXNet and benefits from same syntax and hyperparameter optimisation.
  * Monk Object Detection has a lot of good examples! https://github.com/Tessellate-Imaging/Monk_Object_Detection
  * Monk Image classification https://github.com/Tessellate-Imaging/monk_v1
* Tim Dettmers guide on the latest GPUs. https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/  
