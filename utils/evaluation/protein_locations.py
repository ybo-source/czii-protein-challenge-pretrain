import json

def get coordinates(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    points = data.get("points", [])
    protein_list = []
    for point in points:
        location = point.get("location", {})
        x = location.get("x")
        y = location.get("y")
        z = location.get("z")
        if x is not None and y is not None and z is not None:
            protein_list.append([z/10, y/10, x/10])

    return protein_list