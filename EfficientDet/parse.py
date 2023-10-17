import numpy as np
import json
import xml.etree.ElementTree as ElemTree


def parse_xml(xml_path: str, target: str = "phone"):
    assert target in ("person", "phone")
    tree = ElemTree.parse(xml_path)
    objects = tree.findall("object")
    name = [x.findtext("name") for x in objects]

    result = []
    for i, object_iter in enumerate(objects):
        if ("P" in name[i]) and (target == "phone"):  # phone
            box = object_iter.find("bndbox")  # noqa
            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
            
        elif ("P" not in name[i]) and (target == "person"):  # person
            box = object_iter.find("bndbox")  # noqa
            result.append([int(it.text) for it in box])  # (x1, y1, x2, y2)
            
            
    return result


def parse_prediction(pred: dict, target: str = "phone", threshold: float = 0.0):
    assert target in ("person", "phone")
    result = []
    for v in pred.values():
        if v["score"] < threshold * 100:
            continue

        if ("phone" in v["obj"]) and (target == "phone"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
        elif ("person" in v["obj"]) and (target == "person"):
            box = [v["x1"], v["y1"], v["x2"], v["y2"]]
            result.append(box)
    return result


def parse_distance(json_path: str) -> np.ndarray:
    with open(json_path, "r") as f:
        dist = json.load(f)

    # key = "y_x" , y [0, 480), x [0, 640)
    result = np.zeros((480, 640), dtype=np.float32)
    for k, v in dist.items():
        y, x = k.split("_")
        result[int(y), int(x)] = float(v)
    return result


def mean_distance(dist: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    h, w = dist.shape
    x1 = min(w, max(x1, 0))
    x2 = min(w, max(x2, 0))
    y1 = min(h, max(y1, 0))
    y2 = min(h, max(y2, 0))

    y_diff = y2 - y1
    x_diff = x2 - x1

    pad = 0.4

    crop = dist[y1 + int(y_diff * pad):y1 + int(y_diff * (1 - pad)),
           x1 + int(x_diff * pad):x1 + int(x_diff * (1 - pad))]

    c = np.count_nonzero(crop)
    m = np.sum(crop)
    if c == 0:
        # print("Warning: there is no valid depth point.")
        return 0.0
    return float(m / c)


def center_angle(angle: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    h, w = angle.shape
    x1 = min(w, max(x1, 0))
    x2 = min(w, max(x2, 0))
    y1 = min(h, max(y1, 0))
    y2 = min(h, max(y2, 0))

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    return angle[center_y, center_x]


if __name__ == '__main__':
    parse_distance("data/result/indoor/distance/cal_distance_0_1.json")
    print(parse_xml("data/label/indoor/cal0_1.xml"))
