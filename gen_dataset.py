import geohash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

route = np.array(
    [
        [-73.973695, 40.797041],
        [-73.973651, 40.797149],
        [-73.972259, 40.796556],
        [-73.972713, 40.795936],
        [-73.973006, 40.795531],
        [-73.973207, 40.795253],
        [-73.973555, 40.794804],
        [-73.973724, 40.794586],
        [-73.974186, 40.793962],
        [-73.974635, 40.793332],
        [-73.97509, 40.792705],
        [-73.975552, 40.79207],
        [-73.976005, 40.79144],
        [-73.976449, 40.790814],
        [-73.976923, 40.790174],
        [-73.977374, 40.789563],
        [-73.977863, 40.788887],
        [-73.978373, 40.788199],
        [-73.978829, 40.787567],
        [-73.979294, 40.786939],
        [-73.979762, 40.786293],
        [-73.980223, 40.785659],
        [-73.980678, 40.785031],
        [-73.98117, 40.78435],
        [-73.981687, 40.783671],
        [-73.982129, 40.78303],
        [-73.982595, 40.782395],
        [-73.981176, 40.781802],
        [-73.980982, 40.781719],
        [-73.98073, 40.782439],
        [-73.980696, 40.782522],
        [-73.98063, 40.78266],
    ]
)

if __name__ == "__main__":

    mat_ix = {}

    init_row = "dr5re00"
    current = init_row

    for col in range(128):
        for row in range(128):
            mat_ix[current] = col, row
            current = geohash.neighbors(current)[1]
        init_row = geohash.neighbors(init_row)[5]
        current = init_row


    df = pd.read_csv("data/numeric_routes_dataframe.csv")
    print(df.shape)

    matrices = []

    for route_str in df.route:
        matrix = np.zeros((128, 128))
        route = eval(route_str)
        for point in route:
            h = geohash.encode(point[1], point[0], 7)
            if h in mat_ix:
                matrix[mat_ix[h]] = 255


        #plt.matshow(matrix)
        #plt.show()
        matrices.append(matrix)

    matrices = np.array(matrices)
    np.save('data/routes_matrix_grey.npy', matrices)