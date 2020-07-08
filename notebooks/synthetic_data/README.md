# Synthetic Data tutorial

This tutorial demonstrates training Octopod models on a set of generated color swatches. This is meant to be an easy fast demo of the library's capabilities that can be run on a CPU.

## Dataset
For this tutorial, we develop synthetic image and text data for two tasks:

### Color
- Image: We generate color swatches in blue and green colors by randomly choosing RGB values from the `color_mapping.csv` file.
- Text: We take the complex_color name from the `color_mapping.csv` file to use as the text
- Target: The target variable in this case is the `simple_color` column in `color_mapping.csv` and correpsonds to a classification task labeling rows as `blue` or `green`.
For example, one row would like:

![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/tonks/synthetic_data_tutorial/data/color_swatches/blueberry.jpg)

image: an image patch of RGB values `(70, 65, 150)`

text: `blueberry`

target: `blue`

More example color swatches:

![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/tonks/synthetic_data_tutorial/color_swatch_single.png)

The complex_color names are from a color mapping from XKCD (https://xkcd.com/color/rgb/).

### Pattern
- Image: We generate color swatches of random RGB values. They are either one color (solid) or two (striped).
- Text: We take the name of the file, `striped_0.jpg` as the text for this example. Note: We would never do this in a real model since the target variable is encoded in the text.
- Target: The target variable is the pattern type, either `solid` or `striped`.

For example, one row would look like:

![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/tonks/synthetic_data_tutorial/data/pattern_swatches/striped_7.jpg)

text: `striped_7`

target: `striped`

More example pattern swatches:

![](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/tonks/synthetic_data_tutorial/pattern_swatch_single.png)

## Notebooks
- Step1_generate_data: This is a preliminary notebook where we generate the color swatches using the code in `generate_swatches.py`.
- Step2_train_image_model: In this notebook, we train an image model that can perform both tasks at the same time.
- Step3_train_text_model: In this notebook, we train a text model that can perform both tasks at the same time. 
- Step4: train_ensemble_model: In this notebook, we train an ensemble model by combining our previously trained image and text models.

Note: Notebooks 2 and 3 can be run in either order.
