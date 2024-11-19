import sys
sys.path.append('ariane')
from ariane.read_info import get_netlist_info_dict
from tqdm import tqdm

def get_node_info(pbtxt):
    node_info = {}
    node_info_raw_id_name = {}
    node_cnt = 0
    area_sum = 0.0

    for node in pbtxt.node:
        if node.attr['type'].placeholder.upper() != "MACRO":
            continue
        node_name = node.name
        x = float(node.attr['width'].f)
        y = float(node.attr['height'].f)
        node_info[node_name] = {"id": node_cnt, "x": x, "y": y}
        area_sum += x * y
        if node.attr['type'].placeholder == "MACRO":
            node_info[node_name]["is_hard"] = 1
        else:
            node_info[node_name]["is_hard"] = 0
        node_info_raw_id_name[node_cnt] = node_name
        node_cnt += 1
    print("area_sum = {}".format(area_sum))
    return node_info, node_info_raw_id_name


def get_net_info(pbtxt):
    net_info = {}
    net_name = None
    net_cnt = 0
    pin_cnt = 0
    pin_info = {}
    port_info = {}
    for node in pbtxt.node:
        if node.attr['type'].placeholder.upper() == "MACRO":
            continue
        pin_name = node.name
        if node.attr['type'].placeholder.upper() == "PORT":
            x = float(node.attr['x'].f)
            y = float(node.attr['y'].f)
            port_info[pin_name] = {"x": x, "y": y}
        elif node.attr['type'].placeholder.upper() == "MACRO_PIN":
            macro_name = node.attr['macro_name'].placeholder
            x_offset = float(node.attr['x_offset'].f)
            y_offset = float(node.attr['y_offset'].f)
            pin_info[pin_name] = {"node_name": macro_name, "x_offset": x_offset, "y_offset": y_offset}
            pin_cnt += 1
    print("pin_cnt = {}".format(pin_cnt))
    for node in pbtxt.node:
        net_name = node.name
        if node.attr['type'].placeholder.upper() == "MACRO":
            continue
        net_info[net_name] = {}
        net_info[net_name]["nodes"] = {}
        net_info[net_name]["ports"] = {}
        if 'weight' in node.attr:
            net_info[net_name]["weight"] = float(node.attr['weight'].f)
        else:
            net_info[net_name]["weight"] = 1.0
        for pin_name in node.input:
            if pin_name in port_info:
                assert pin_name not in net_info[net_name]["ports"]
                net_info[net_name]["ports"][pin_name] = {}
                net_info[net_name]["ports"][pin_name]["x"] = port_info[pin_name]["x"]
                net_info[net_name]["ports"][pin_name]["y"] = port_info[pin_name]["y"]
            elif pin_name in pin_info:
                node_name = pin_info[pin_name]["node_name"]
                if node_name in net_info[net_name]["nodes"]:
                    if "x_offsets" not in net_info[net_name]["nodes"][node_name]:
                        net_info[net_name]["nodes"][node_name]["x_offsets"] = [net_info[net_name]["nodes"][node_name]["x_offset"]]
                        net_info[net_name]["nodes"][node_name]["y_offsets"] = [net_info[net_name]["nodes"][node_name]["y_offset"]]
                    net_info[net_name]["nodes"][node_name]["x_offsets"].append(pin_info[pin_name]["x_offset"])
                    net_info[net_name]["nodes"][node_name]["y_offsets"].append(pin_info[pin_name]["y_offset"])
                net_info[net_name]["nodes"][node_name] = {}
                net_info[net_name]["nodes"][node_name]["x_offset"] = pin_info[pin_name]["x_offset"]
                net_info[net_name]["nodes"][node_name]["y_offset"] = pin_info[pin_name]["y_offset"]
            else:
                assert False
        out_pin_name = net_name
        if out_pin_name in port_info:
            assert out_pin_name not in net_info[net_name]["ports"]
            net_info[net_name]["ports"][out_pin_name] = {}
            net_info[net_name]["ports"][out_pin_name]["x"] = port_info[out_pin_name]["x"]
            net_info[net_name]["ports"][out_pin_name]["y"] = port_info[out_pin_name]["y"]
        elif out_pin_name in pin_info:
            node_name = pin_info[out_pin_name]["node_name"]
            assert node_name not in net_info[net_name]["nodes"]
            net_info[net_name]["nodes"][node_name] = {}
            net_info[net_name]["nodes"][node_name]["x_offset"] = pin_info[out_pin_name]["x_offset"]
            net_info[net_name]["nodes"][node_name]["y_offset"] = pin_info[out_pin_name]["y_offset"]
        else:
            print("out_pin_name = {}".format(out_pin_name))
            assert False

    for net_name in list(net_info.keys()):
        if len(net_info[net_name]["nodes"]) + \
            len(net_info[net_name]["ports"]) <= 1:
            net_info.pop(net_name)
    for net_name in net_info:
        net_info[net_name]['id'] = net_cnt
        net_cnt += 1
    print("adjust net size = {}".format(len(net_info)))
    return net_info, port_info

                
def main():
    path = 'ariane/netlist.pb.txt'
    pbtxt = get_netlist_info_dict(path)
    node_info = get_node_info(pbtxt)
    net_info, port_info = get_net_info(pbtxt)


if __name__ == "__main__":
    main()
