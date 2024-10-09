# CLIP Aesthetic Image Selector

## Overview
This project provides a script that uses the CLIP model to select the most aesthetically pleasing images from a given directory. It uses a linear layer to predict the aesthetic score for each image. The selected images are then grouped based on orientation, and the results are saved in organized subfolders.

## Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended)

## Installation

1. **Clone the Repository**:
   ```sh
   git clone <repository_url>
   cd select_best_images
   ```

2. **Install Dependencies**: Install the required Python libraries using `pip`:

   ```sh
   python -m pip install -r requirements.txt
   ```

   The `requirements.txt` includes the following main dependencies:

   - `torch`
   - `transformers`
   - `safetensors`
   - `scikit-learn`
   - `Pillow`

3. **Download Pretrained Weights**: Ensure you have downloaded the `open_clip_pytorch_model.safetensors` file. Place it in the same directory as the script.

## Usage

To run the script, use the following command:

```sh
python select_best_images.py <group_size> <number_of_images> <path_to_images>
```

- **`group_size`**: Number of images per group for organizing output.
- **`number_of_images`**: Number of top images to select from the provided directory.
- **`path_to_images`**: Path to the folder containing the images.

Example:

```sh
python select_best_images.py 10 30 social/20240922_2nd_KC_Memorial_Poker_Run/
```
This command will select the top 30 images from the given folder and create subfolders under `best/` in groups of 10.

## Script Workflow

1. **Load Models**: The script loads the CLIP model and an aesthetic scoring head. The aesthetic head is a simple linear layer that takes the CLIP model's image features and outputs an aesthetic score.

2. **Calculate Aesthetic Scores**: Each image is processed, and an aesthetic score is computed using the pretrained model. Scores are cached in a `rating.json` file to avoid recalculating for previously processed images.

3. **Clustering for Diversity**: The script uses `KMeans` clustering to ensure that the selected images are diverse. This helps prevent multiple similar-looking images from being selected.

4. **Sorting and Grouping**: After scoring, images are sorted based on their scores. Selected images are divided into two categories based on their orientation (landscape or portrait) and are saved in subfolders of the `best/` directory. Images are grouped according to the specified `group_size`, ensuring each group only contains images of the same orientation.

5. **Save Scores**: The calculated scores are saved in a `rating.json` file within the images folder. If the file already exists, the script will load the scores from it and only calculate scores for new images.

## Output

- The best selected images are saved in a subdirectory called `best/` within the input image folder.
- Images are grouped by their orientation (landscape or portrait), with each group having a folder named after its group number.
- A `rating.json` file is saved, containing the image filenames and their corresponding scores.

## Example Output Structure

If you run the script with `group_size=10` and `number_of_images=30`, the resulting folder structure might look like this:

```
social/20240922_2nd_KC_Memorial_Poker_Run/
    ├── best/
    │   ├── 1/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── 2/
    │   └── 3/
    ├── rating.json
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## Notes

- The script works better with a GPU to speed up the inference process, but it can also run on CPU (albeit slower).
- If `rating.json` already exists, the script will reuse existing scores, making subsequent runs faster.

## License

This project is licensed under the MIT License.
