import ast
import os
from pathlib import Path
from random import randint

import numpy as np
from PIL import Image
import pandas as pd


def generate_color_swatches(
    output_folder='data/color_swatches',
    output_csv='color_dataset.csv',
    select_colors=['blue', 'green'],
    swatch_size=(100, 100)
):
    """
    Generate synthetic color swatches based on a mapping between complex color names, simple color names, and RGB values.

    Parameters
    ----------
    output_folder: str or Path (defaults to 'data/color_swatches/')
        location to store color swatches
    output_csv: str (defaults to 'color_dataset.csv')
        location to store csv containing information about color swatches
    select_colors: list (defaults to `['blue', 'green']`) 
        subset of colors to take from `color_mapping.csv`
    swatch_size: tuple (defaults to `(100, 100)`)
        size of swatches to be generated
    
    Side Effects
    ------------
    - Writes jpg files of color swatches to specified location
    - Writes csv file containing info about swatches to specified location
    """
    df = pd.read_csv('color_mapping.csv')
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    final_df = df[df.simple_color.isin(select_colors)]

    image_locs = []
    for row in final_df.iterrows():
        complex_color = row[1]['complex_color'].replace(' ', '_')
        simple_color = row[1]['simple_color']
        rgb = row[1]['rgb']
        rgb = ast.literal_eval(str(rgb))

        img = Image.new('RGB', swatch_size, rgb)

        image_loc = output_folder / f'{complex_color}.jpg'
        try:
            img.save(image_loc)
        except FileNotFoundError:
            # We'll skip these errors as they probably arise from something weird in the color name.
            image_loc = None
        image_locs.append(image_loc)

    final_df['image_locs'] = image_locs
    final_df = final_df[final_df['image_locs'].notnull()]

    final_df.to_csv(output_folder / output_csv, index=False)


def generate_pattern_swatches(    
    output_folder='data/pattern_swatches',
    output_csv='pattern_dataset.csv',
    swatch_size=(100, 100)
):
    """
    Generate 50 synthetic pattern swatches of either solid or striped pattern.

    Parameters
    ----------
    output_folder: str (defaults to 'data/pattern_swatches/')
        location to store pattern swatches
    output_csv: str (defaults to 'pattern_dataset.csv')
        location to store csv containing information about pattern swatches
    swatch_size: tuple (defaults to `(100, 100)`)
        size of swatches to be generated
    
    Side Effects
    ------------
    - Writes jpg files of pattern swatches to specified location
    - Writes csv file containing info about swatches to specified location
    """
    cwd = os.getcwd()
    outdir = cwd / Path(output_folder)

    if not outdir.exists():
        outdir.mkdir(parents=True)

    image_locs = []
    pattern_types = []
    for i in range(50):
        img = generate_random_patched_color_swatch(
            num_images=1,
            percent_rgb_2=0.5,
            size=swatch_size
        )[0]
        
        image_loc = outdir / f'striped_{i}.jpg'
        img.save(image_loc)
        image_locs.append(image_loc)
        pattern_types.append('striped')

    for j in range(50):
        img = generate_random_solid_color_swatch(num_images=1, size=swatch_size)[0]
        
        image_loc = outdir / f'solid_{j}.jpg'
        img.save(image_loc)
        image_locs.append(image_loc)
        pattern_types.append('solid')

    pattern_df = pd.DataFrame({
        'pattern_type': pattern_types,
        'image_locs': image_locs
    })

    pattern_df.to_csv(outdir / output_csv, index=False)


def get_random_color():
    return tuple([randint(0, 255) for i in range(0, 3)])


def generate_color_swatch(
    rgb_values,
    percents,
    size
):
    """
    Creates multi colored image with each rgb value in rgb_values present in
    the percentage from percents at the same index. The number of pixels is
    rounded to the nearest whole number meaning the percentage may not be exactly
    the same.

    Parameters
    ----------
    rgb_values: array of tuples
        array of rgb values, e.g. [(255, 255, 255), (0, 255, 0)]
    percent_rgb: array of floats
        The percentage of pixels each color value.
        Number of actual pixels will be rounded to the nearest
        int if it does not equate to whole number.
        Should sum to 1.0
        e.g. [.25, .75]
    size: tuple of ints
        width and height of image

    Returns
    ------------
    Generated PIL.Image object
    """
    if len(rgb_values) != len(percents):
        error_message = 'You must send a percent for each color value'
        raise ValueError(error_message)
    if not np.isclose([np.sum(percents)], [1.], atol=1e-03)[0]:
        error_message = 'Percentages must add to 1.0 when generating swatches'
        raise ValueError(error_message)
    else:
        total_pixels = size[0]*size[1]

        numb_rgb = [round(total_pixels * percent) for percent in percents]

        img_data = _generate_patched_pixel_data(rgb_values, numb_rgb, total_pixels)

        img = Image.new('RGB', size)
        img.putdata(img_data)

        return img


def _generate_patched_pixel_data(
    rgb_values,
    num_values,
    total_pixels
):
    """
    Creates an array of rgb values with the number of entries equivalent
    to total_pixels. Color split is determined by num_rgb_2.
    (# of rgb_value_2 entries = num_rgb_2)

    Parameters
    ----------
    rgb_values: array of tuples
        array of rgb values, e.g. [(255, 255, 255), (0, 255, 0)]
    num_values: array
        an array with the number of pixels for each value
        e.g. [10, 20]
    total_pixels: int
        the total number of pixels desired in the new image

    Returns
    ------------
    Array of tuples. Each entry is the rgb value of a pixel.
    """
    img_data = np.repeat(rgb_values, num_values, axis=0)

    # If rounding means there are too many of too few pixels add/remove them
    while len(img_data) < total_pixels:
        img_data.append(rgb_values[0])
    while len(img_data) > total_pixels:
        img_data.pop()

    img_data = [tuple(i for i in rgb) for rgb in img_data]

    return img_data


def generate_random_patched_color_swatch(num_images=1, percent_rgb_2=.25, size=(60, 30)):
    """
    Creates a multi color image that consists of two randomly generated rgb values.
    Color two is percent_rgb_2 percentage of total pixels. This value may
    be rounded to make the number of pixels of rgb_value_2 a whole number.

    Parameters
    ----------
    num_images: int (defaults to `1` )
        number of images to create
    percent_rgb_2: float (defaults to `0.25`)
        The percentage of pixels that should be rgb_value_2.
        Number of actual pixels will be rounded to the nearest
        int if it does not equate to whole number.
    size: tuple of ints (defaults to `(60, 30)`)
        width and height of image

    Returns
    ------------
    list of generated PIL.Image objects
    """
    patched_images = []
    for i in range(0, num_images):
        img = generate_color_swatch(
            [get_random_color(), get_random_color()],
            [1. - percent_rgb_2, percent_rgb_2],
            size
        )
        patched_images.append(img)

    return patched_images


def generate_random_solid_color_swatch(num_images=1, size=(60, 30)):
    """
    Generates solid color swatches of randomly selected colors.

    Parameters
    ----------
    num_images: int (defaults to `1`)
        number of images to create
    size: tuple of ints (defaults to `(60, 30)`)
        width and height of image

    Returns
    ------------
    List of generated PIL.Image objects
    """
    solid_swatches = []
    for i in range(0, num_images):
        rgb_value = get_random_color()
        img = Image.new('RGB', size, color=rgb_value)
        solid_swatches.append(img)

    return solid_swatches
