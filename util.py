
# util function to flatten data to 1D
def flatten(array):
    return [item for sublist in array for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]