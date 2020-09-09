# Object-Detection

For jupyter notebook use follow the following commands to set up your environment, Check versions from requirements.txt.

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

Steps 5 to 16 with the respective versions should be:

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
