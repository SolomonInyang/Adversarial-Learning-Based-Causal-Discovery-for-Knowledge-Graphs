import numpy as np

import tensorly as tl
from tensorly import tenalg

from tqdm import tqdm

import lingam
from lingam.utils import make_dot

import time

# from dag_construction import construct_dag


class Q_Mat:
    def __init__(self, dataset, algorithm, plot):
        self.dataset = dataset
        self.algorithm = algorithm
        self.plot = plot

    def initialize(self):
        n_dims = [5, 7, 10]
        Q_matrices = []

        # load embeddings from files
        for i in n_dims:
            path_to_embeddings = "tucker_" + str(i)
            e_path = self.dataset + "/" + path_to_embeddings + "/ent_embedding.tsv"
            w_path = self.dataset + "/" + path_to_embeddings + "/W.tsv"
            r_path = self.dataset + "/" + path_to_embeddings + "/rel_embedding.tsv"

            e = np.loadtxt(fname=e_path, delimiter="\t", skiprows=1)
            w = np.loadtxt(fname=w_path, delimiter="\t", skiprows=0)
            r = np.loadtxt(fname=r_path, delimiter="\t", skiprows=0)

            x = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]

            w = w.reshape(x, x, x)

            m_1 = tl.tenalg.mode_dot(w, e, 1, transpose=False)

        # Generate Q matrix
        Q_tensor = []

        if self.dataset == "fb15k-237":
            n_r = 237
        else:
            n_r = 11

        print(f"Generating Q matrix for dim = {i}")

        for i in tqdm(range(n_r)):
            m_2 = tl.tenalg.mode_dot(m_1, r[i], 2, transpose=False)
            m_3 = np.dot(m_2, e)
            Q_tensor.append(m_3)

        Q_tensor = np.array(Q_tensor)

        Q_matrix = Q_tensor.reshape(
            Q_tensor.shape[0], Q_tensor.shape[1] * Q_tensor.shape[2]
        )

        Q_matrices.append(Q_matrix)

        print("The Q matrices have been computed for dimensions 5, 7 and 10.")
        print(Q_matrices)
        return Q_matrices

    def DirectLiNGAM_test(self, Q_matrix):
        if self.algorithm == "DirectLiNGAM":
            model = lingam.DirectLiNGAM()
        else:
            model = lingam.ICALiNGAM()

        start_time = time.time()
        model.fit(Q_matrix)
        total_time = time.time() - start_time

        causal_order = model.causal_order_

        p_value = model.get_error_independence_p_values(Q_matrix)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        print(f"total time taken to execute : {total_time}")
        print(f"causal order : {causal_order}")
        print(f"mean p-value is : {np.mean(p_value)}")

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")

        print("Initializing file write sequence...")

        f = open("results.txt", "a")
        f.write("Dataset: " + self.dataset)
        f.write(" Algorithm: " + self.algorithm)
        f.write(
            " "
            + str(total_time)
            + " "
            + str(np.mean(p_value))
            + " "
            + str(causal_order)
        )
        f.write("\n")
        f.close()

        print("Done!")

        if self.plot:
            print("Initializing plot...")
            name = self.dataset + "_" + self.algorithm + "_" + str(Q_matrix.shape[1])
            dot = make_dot(model.adjacency_matrix_)
            dot.format = "png"
            dot.render(name)
            print("Done!")

            # print("Initializing plot...")
            # construct_dag(model.adjacency_matrix_)
            # print("Done!")

