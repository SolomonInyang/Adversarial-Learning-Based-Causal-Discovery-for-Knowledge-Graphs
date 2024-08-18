import networkx as nx
import numpy as np
from sklearn.preprocessing import scale

import pandas as pd
from pandas import DataFrame, Series

from joblib import Parallel, delayed

from settings import SETTINGS


class GraphSkeletonModel(object):
    """Base class for undirected graph recovery directly out of data."""

    def __init__(self):
        """Init the model."""
        super(GraphSkeletonModel, self).__init__()

    def predict(self, data):
        """Infer a undirected graph out of data.

        Args:
            data (pandas.DataFrame): observational data

        Returns:
            networkx.Graph: Graph skeleton

        .. warning::
           Not implemented. Implemented by the algorithms.
        """
        raise NotImplementedError


class FeatureSelectionModel(GraphSkeletonModel):
    """Base class for methods using feature selection
    on each variable independently.
    """

    def __init__(self):
        """Init the model."""
        super(FeatureSelectionModel, self).__init__()

    def predict_features(self, df_features, df_target, idx=0, **kwargs):
        """For one variable, predict its neighbouring nodes.

        Args:
            df_features (pandas.DataFrame):
            df_target (pandas.Series):
            idx (int): (optional) for printing purposes
            kwargs (dict): additional options for algorithms

        Returns:
            list: scores of each feature relatively to the target

        .. warning::
           Not implemented. Implemented by the algorithms.
        """
        raise NotImplementedError

    def run_feature_selection(self, df_data, target, idx=0, **kwargs):
        """Run feature selection for one node: wrapper around
        ``self.predict_features``.

        Args:
            df_data (pandas.DataFrame): All the observational data
            target (str): Name of the target variable
            idx (int): (optional) For printing purposes

        Returns:
            list: scores of each feature relatively to the target
        """
        list_features = list(df_data.columns.values)
        list_features.remove(target)
        df_target = pd.DataFrame(df_data[target], columns=[target])
        df_features = df_data[list_features]

        return self.predict_features(df_features, df_target, idx=idx, **kwargs)

    def predict(self, df_data, threshold=0.05, **kwargs):
        """Predict the skeleton of the graph from raw data.

        Returns iteratively the feature selection algorithm on each node.

        Args:
            df_data (pandas.DataFrame): data to construct a graph from
            threshold (float): cutoff value for feature selection scores
            kwargs (dict): additional arguments for algorithms

        Returns:
            networkx.Graph: predicted skeleton of the graph.
        """
        njobs = kwargs.get("njobs", SETTINGS.NJOBS)
        list_nodes = list(df_data.columns.values)
        if njobs != 1:
            result_feature_selection = Parallel(n_jobs=njobs)(delayed(self.run_feature_selection)
                                                              (df_data, node, idx, **kwargs)
                                                              for idx, node in enumerate(list_nodes))
        else:
            result_feature_selection = [self.run_feature_selection(df_data, node,
                                                                   idx, **kwargs)
                                        for idx, node in enumerate(list_nodes)]
        for idx, i in enumerate(result_feature_selection):
            try:
                i.insert(idx, 0)
            except AttributeError:  # if results are numpy arrays
                result_feature_selection[idx] = np.insert(i, idx, 0)
        matrix_results = np.array(result_feature_selection)
        matrix_results *= matrix_results.transpose()
        np.fill_diagonal(matrix_results, 0)
        matrix_results /= 2

        graph = nx.Graph()

        for (i, j), x in np.ndenumerate(matrix_results):
            if matrix_results[i, j] > threshold:
                graph.add_edge(list_nodes[i], list_nodes[j],
                               weight=matrix_results[i, j])
        for node in list_nodes:
            if node not in graph.nodes():
                graph.add_node(node)
        return graph


class PairwiseModel(object):
    """Base class for all pairwise causal inference models

    Usage for undirected/directed graphs and CEPC df format.
    """

    def __init__(self):
        """Init."""
        super(PairwiseModel, self).__init__()

    def predict(self, x, *args, **kwargs):
        """Generic predict method, chooses which subfunction to use for a more
        suited.

        Depending on the type of `x` and of `*args`, this function process to execute
        different functions in the priority order:

        1. If ``args[0]`` is a ``networkx.(Di)Graph``, then ``self.orient_graph`` is executed.
        2. If ``args[0]`` exists, then ``self.predict_proba`` is executed.
        3. If ``x`` is a ``pandas.DataFrame``, then ``self.predict_dataset`` is executed.
        4. If ``x`` is a ``pandas.Series``, then ``self.predict_proba`` is executed.

        Args:
            x (numpy.array or pandas.DataFrame or pandas.Series): First variable or dataset.
            args (numpy.array or networkx.Graph): graph or second variable.

        Returns:
            pandas.Dataframe or networkx.Digraph: predictions output
        """
        if len(args) > 0:
            if type(args[0]) == nx.Graph or type(args[0]) == nx.DiGraph:
                return self.orient_graph(x, *args, **kwargs)
            else:
                y = args.pop(0)
                return self.predict_proba((x, y), *args, **kwargs)
        elif type(x) == DataFrame:
            return self.predict_dataset(x, *args, **kwargs)
        elif type(x) == Series:
            return self.predict_proba((x.iloc[0], x.iloc[1]), *args, **kwargs)

    def predict_proba(self, dataset, idx=0, **kwargs):
        """Prediction method for pairwise causal inference.

        predict_proba is meant to be overridden in all subclasses

        Args:
            dataset (tuple): Couple of np.ndarray variables to classify
            idx (int): (optional) index number for printing purposes

        Returns:
            float: Causation score (Value : 1 if a->b and -1 if b->a)
        """
        raise NotImplementedError

    def predict_dataset(self, x, **kwargs):
        """Generic dataset prediction function.

        Runs the score independently on all pairs.

        Args:
            x (pandas.DataFrame): a CEPC format Dataframe.
            kwargs (dict): additional arguments for the algorithms

        Returns:
            pandas.DataFrame: a Dataframe with the predictions.
        """
        printout = kwargs.get("printout", None)
        pred = []
        res = []
        x.columns = ["A", "B"]
        for idx, row in x.iterrows():
            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba((a, b), idx=idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1]])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, graph, printout=None, **kwargs):
        """Orient an undirected graph using the pairwise method defined by the subclass.

        The pairwise method is ran on every undirected edge.

        Args:
            df_data (pandas.DataFrame): Data
            graph (networkx.Graph): Graph to orient
            printout (str): (optional) Path to file where to save temporary results

        Returns:
            networkx.DiGraph: a directed graph, which might contain cycles

        .. warning::
           Requirement : Name of the nodes in the graph correspond to name of
           the variables in df_data
        """
        if isinstance(graph, nx.DiGraph):
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif isinstance(graph, nx.Graph):
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []

        for idx, (a, b) in enumerate(edges):
            weight = self.predict_proba(
                (df_data[a].values.reshape((-1, 1)),
                 df_data[b].values.reshape((-1, 1))), idx=idx,
                **kwargs)
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            elif weight < 0:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output


class GraphModel(object):
    """Base class for all graph causal inference models.

    Usage for undirected/directed graphs and raw data. All causal discovery
    models out of observational data base themselves on this class. Its main
    feature is the predict function that executes a function according to the
    given arguments.
    """

    def __init__(self):
        """initialize."""
        super(GraphModel, self).__init__()

    def predict(self, df_data, graph=None, **kwargs):
        """Orient a graph using the method defined by the arguments.

        Depending on the type of `graph`, this function process to execute
        different functions:

        1. If ``graph`` is a ``networkx.DiGraph``, then ``self.orient_directed_graph`` is executed.
        2. If ``graph`` is a ``networkx.Graph``, then ``self.orient_undirected_graph`` is executed.
        3. If ``graph`` is a ``None``, then ``self.create_graph_from_data`` is executed.

        Args:
            df_data (pandas.DataFrame): DataFrame containing the observational data.
            graph (networkx.DiGraph or networkx.Graph or None): Prior knowledge on the causal graph.

        .. warning::
            Requirement : Name of the nodes in the graph must correspond to the
            name of the variables in df_data
        """

        if graph is None:
            return self.create_graph_from_data(df_data, **kwargs)
        elif isinstance(graph, nx.DiGraph):
            return self.orient_directed_graph(df_data, graph, **kwargs)
        elif isinstance(graph, nx.Graph):
            return self.orient_undirected_graph(df_data, graph, **kwargs)
        else:
            print('Unknown Graph type')
            raise ValueError

    def orient_undirected_graph(self, data, umg, **kwargs):
        """Orient an undirected graph.

        .. note::
            Not implemented: will be implemented by the model classes.
        """

        raise NotImplementedError

    def orient_directed_graph(self, data, dag, **kwargs):
        """Re/Orient an undirected graph.

        .. note::
           Not implemented: will be implemented by the model classes.
        """

        raise NotImplementedError

    def create_graph_from_data(self, data, **kwargs):
        """Infer a directed graph out of data.

        .. note::
            Not implemented: will be implemented by the model classes.
        """

        raise NotImplementedError
