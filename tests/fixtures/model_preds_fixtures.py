import pytest
import torch


@pytest.fixture()
def test_label_dictionary():
    color_list = ['beige', 'black']
    dress_length_list = ['mini', 'maxi']
    pattern_list = ['dotted', 'floral']
    sleeve_length_list = ['sleeveless', 'long_sleeve']
    attribute_label_dict = {
        'color': color_list,
        'dress_length': dress_length_list,
        'pattern': pattern_list,
        'sleeve_length': sleeve_length_list
    }

    return attribute_label_dict


@pytest.fixture()
def test_raw_logits():
    return {
        'color': torch.tensor([[0, 5.0]]),
        'dress_length': torch.tensor([[0, 5.0]]),
        'pattern': torch.tensor([[5.0, 0]]),
        'sleeve_length': torch.tensor([[5.0, 0]])
    }
