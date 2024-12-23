
# util function to flatten data to 1D


# param: 2D list
# iterating over multi dims becomes inefficient in higher dimensions
# as such, we use a cartesian product to do this recursively
# [[0], [0,1,2], [2]] => [[0,0,2], [0,1,2], [0,2,2]]
# consider the matrix A which is nxn
# A[1,:] = [A[1,0], ... , A[1,n]]
# intuitively, a slice and index make sense because now we are getting
# all elements from the slice for that given index
def cartesian_product(array):
    if not array:
        return [[]]

    rest = (cartesian_product(array[1:]))

    res = []
    for start_item in array[0]:
        for remaining_item in rest:
            res.append([start_item] + remaining_item)

    return res


class ShapeUtils():

    @staticmethod
    def flatten(array):
        return [item for sublist in array for item in (ShapeUtils.flatten(sublist) if isinstance(sublist, list) else [sublist])]

    @staticmethod
    def get_shape(data, device):
        if isinstance(data, list):
            shape = []
            current_level = data
            while isinstance(current_level, list):
                shape.append(len(current_level))
                current_level = current_level[0] if len(current_level) > 0 else []
            return shape
        else:
            raise ValueError("tensor data must be a list")

    @staticmethod
    def create_data_struct(tensor, array, shape):
        return tensor.initialize(ShapeUtils.flatten(array), shape)
