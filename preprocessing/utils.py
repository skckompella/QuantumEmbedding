
def write_data_to_file(data, file_name, join=False):

    with open(file_name, "w") as fp:
        for line in data:
            if join:
                fp.write(" ".join(line) + "\n")
            else:
                fp.write(line + "\n")
