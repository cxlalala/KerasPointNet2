import os

# Set up training datasets
def open_file_list(directory, name):
    full_file_path = os.path.join(directory, name)
    try:
        training_list = open(full_file_path, "r")
        return [os.path.join(directory, line[:-1]) for line in training_list]
    except:
        print("Expected a training list at {}".format(full_file_path))
        exit(-1)
