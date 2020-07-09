![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/tonks/fashion_data_tutorial/fashion_samples.png)

# Fashion Data Tutorial

This tutorial demonstrates training Octopod models on an open source fashion dataset consisting of images and text descriptions.

This set of notebooks was run on an AWS `p3.2xlarge` machine. 

## Dataset
For this tutorial, we use a Kaggle dataset of fashion images. The dataset can be found here:  https://www.kaggle.com/paramaggarwal/fashion-product-images-small.

This tutorial is much closer to how we actually use Octopod at ShopRunner.

The dataset consists of product images and names as well as a number of attributes for each product.

For this tutorial, we used the `gender` and `season` attributes as our two tasks. To simulate having two different datasets, we sub-sampled the data once for each of the attributes.

## Main Notebooks
- Step1_format_data: In this notebook, we format the data so that it is ready for Octopod.
- Step2_train_image_model: In this notebook, we train an image model that can perform both tasks at the same time.
- Step3_train_text_model: In this notebook, we train a text model that can perform both tasks at the same time. 
- Step4_train_ensemble_model: In this notebook, we train an ensemble model by combining our previously trained image and text models.

Note: Notebooks 2 and 3 can be run in either order.

## Optional Notebooks
If desired, we can train two separate ResNets for the two tasks. You might do this if you think the tasks are destructively interferring with one another. See https://dawn.cs.stanford.edu/2019/03/22/glue/ and https://arxiv.org/pdf/1807.06708.pdf section 3.1 for more info.
- Optional_Step5_train_gender_image_model: In this notebook, we train a gender image model.
- Optional_Step6_train_season_image_model: In this notebook, we train a season image model.
- Optional_Step7_train_ensemble_model_with_two_resnets: In this notebook, we train an ensemble model using the text model trained in Step3 and the two single task ResNets.
