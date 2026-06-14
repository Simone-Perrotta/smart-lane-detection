import cv2

from src.smart_lane_detection.lane_detector import detect_lanes

import cv2

from src.smart_lane_detection.lane_detector import detect_lanes


def main() -> None:
    """
    Run lane detection on a sample image, save the result,
    and display it on screen.
    """

    # Load the input image from the project data folder.
    input_path = "data/images/road.jpg"

    # Define where the processed image will be saved.
    output_path = "results/images/road_output.jpg"

    # Read the image using OpenCV.
    image = cv2.imread(input_path)

    # If OpenCV cannot read the image, stop the program with a clear error.
    if image is None:
        raise FileNotFoundError(f"Could not read image from path: {input_path}")

    # Run the lane detection pipeline.
    lane_image = detect_lanes(image)

    # Save the processed image so we can inspect it later.
    cv2.imwrite(output_path, lane_image)

    # Display the output image with detected lanes.
    cv2.imshow("Lane Detection", lane_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Lane detection completed. Output saved to: {output_path}")


if __name__ == "__main__":
    main()