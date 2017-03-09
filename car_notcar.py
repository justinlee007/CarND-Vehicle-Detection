import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from trainer import *


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)

    print("car_list[0]={}".format(car_list[0]))

    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    # Return data_dict
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing car/not-car pair")
    parser.add_argument("-show", action="store_true", help="Show first car/not-car pair")
    parser.add_argument("-save", action="store_true", help="Save car/not-car pair image to disk")
    results = parser.parse_args()
    show = bool(results.show)
    save = bool(results.save)
    cars, notcars = load_features()
    data_info = data_look(cars, notcars)

    print("Your function returned a count of",
          data_info["n_cars"], " cars and",
          data_info["n_notcars"], " non-cars")
    print("of size: ", data_info["image_shape"], " and data type:",
          data_info["data_type"])
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Plot the examples
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(car_image)
    ax1.set_title("Example Car Image", fontsize=24)
    ax2.imshow(notcar_image)
    ax2.set_title("Example Not-car Image", fontsize=24)
    if show:
        plt.show()
    if save:
        save_file_name = "car_not_car.png"
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")