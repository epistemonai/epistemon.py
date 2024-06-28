
def flatten_array(arr: list) -> list:
    flattened = []
    for i in arr:
        if isinstance(i, list): flattened.extend(flatten_array(i))
        else: flattened.append(i)
    return flattened

def array_shape(arr: list) -> tuple:
    shape = []
    while isinstance(arr, list):
        shape.append(len(arr))
        arr = arr[0]
    return tuple(shape)