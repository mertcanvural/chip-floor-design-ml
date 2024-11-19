# import tensorflow as tf

from google.protobuf import text_format

import laiyao_pb2


def load_pbtxt_file(path):
    """Read .pbtxt file.
    
    Args: 
        path: Path to StringIntLabelMap proto text file (.pbtxt file).
        
    Returns:
        A StringIntLabelMapProto.
        
    Raises:
        ValueError: If path is not exist.
    """
    # if not tf.gfile.Exists(path):
    #     raise ValueError('`path` is not exist.')
        
    # with tf.gfile.GFile(path, 'r') as fid:
    #     pbtxt_string = fid.read()
    #     pbtxt = laiyao_pb2.StudentInfo()
    #     try:
    #         text_format.Merge(pbtxt_string, pbtxt)
    #     except text_format.ParseError:
    #         pbtxt.ParseFromString(pbtxt_string)
    fid = open(path, 'r')
    pbtxt_string = fid.read()
    pbtxt = laiyao_pb2.GraphDef()
    try:
        text_format.Merge(pbtxt_string, pbtxt)
    except text_format.ParseError:
        pbtxt.ParseFromString(pbtxt_string)
    return pbtxt


def get_netlist_info_dict(path):
    """Reads a .pbtxt file and returns a dictionary.
    
    Args:
        path: Path to StringIntLabelMap proto text file.
        
    Returns:
        A dictionary mapping class names to indices.
    """
    pbtxt = load_pbtxt_file(path)
    
    # result_dict = {}
    # for node in pbtxt.node:
    #     print("node_name: {}".format(node.name))
    return pbtxt


def main():
    get_netlist_info_dict('netlist.pb.txt')


if __name__ == "__main__":
    main()