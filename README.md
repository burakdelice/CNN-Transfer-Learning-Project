# Hough Circle Detection

## Overview
This Python script detects circles in images using the Hough Circle Transformation technique. It processes images from input folders, identifies circles, and saves the results in corresponding output folders.

## Requirements
- Python 3.x
- OpenCV (cv2)
- NumPy

## Usage
1. **Installation:**
   - Ensure Python is installed on your system.
   - Install required libraries using pip:
     ```
     pip install opencv-python numpy
     ```

2. **Setup:**
   - Organize your images into separate folders:
     - *Train*: Contains training images.
     - *TestV*: Contains validation/testing images.
     - *TestR*: Contains testing images (for a different scenario, perhaps).
   - Ensure each folder contains only image files.

3. **Execution:**
   - Run the script `hough_circle_detection.py`.
   - Detected circles will be saved in corresponding output folders (*Train_Hough*, *TestV_Hough*, *TestR_Hough*).

## Parameters
- `minDist`: Minimum distance between detected circles. Adjust for accuracy.
- `param1`: Upper threshold for the edge detector.
- `param2`: Threshold for circle detection. Lower values detect more circles.
- `minRadius`: Minimum radius of the detected circles.
- `maxRadius`: Maximum radius of the detected circles.

## Output
- Detected circles are outlined in green on the original images.
- Results are saved with the same filenames in their respective output folders.

## Note
- Ensure input folders (*Train*, *TestV*, *TestR*) exist in the same directory as the script.
- Output folders (*Train_Hough*, *TestV_Hough*, *TestR_Hough*) will be created automatically if they don't exist.
- Adjust parameters based on image quality and circle characteristics for optimal results.

## Example
- For each processed image, the script prints the filename and the detection outcome.

## Conclusion
- Upon completion, the script prints "Process completed."

## Contributors
- Developed by [Burak Delice] ([burakdelice/https://github.com/burakdelice]).

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)

## License
- This project is licensed under the [MIT License](LICENSE).

Feel free to modify and use this script according to your needs. If you encounter any issues or have suggestions for improvement, please don't hesitate to reach out.
