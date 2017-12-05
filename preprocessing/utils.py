
def write_data_to_file(data, file_name, join=False):
    """

    :param data: some data in the form of a list
    :param file_name: file where the above data has to be stored
    :param join: if the list is single or double dimensional array
    :return:
    """
    with open(file_name, "w") as fp:
        for line in data:
            if join:
                fp.write(" ".join(line) + "\n")
            else:
                fp.write(line + "\n")
