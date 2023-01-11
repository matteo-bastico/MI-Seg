import json

from pathlib import Path
from monai.config import PathLike
from monai.data.decathlon_datalist import _append_paths
from typing import Dict, List, Optional


def load_decathlon_datalist_with_modality(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir: Optional[PathLike] = None,
) -> List[Dict]:
    """Load image/label paths of decathlon challenge from JSON file including data modality as

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0, 'modality':'CT'},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1, 'modality':'MRI'}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    # Add modality of data
    for data in expected_data:
        data['modality'] = json_data['modality']
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)