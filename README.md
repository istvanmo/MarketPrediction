Prediction of Direction of Stock Market Index Movement Using Hybrid GA-ANN
=============================

This is an implementation of this article: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155133#sec014

  The modifications:
    - You can change the training between Only GA, Only Backpropagation and Hybrid GA-BP.

Dependencies
------------

- **Tensorflow**
- **Keras**
- **numpy**
- **skimage**


Usage
-----

- You should run the "Main_GA.py" with the desired options to train the model.
    - Only Backpropagation -> "only_bp" should be True (in this case "is_bp" is irrelevant)
    - Only Genetic Algorithm -> "only_bp" should be False and "is_bp" should be False too
    - Hybrid -> "only_bp" should be False and "is_bp" should be True

Notes
-----
  - Model save option is not added yet!
