import numpy as np
import time
import os

from multiprocessing import Pool
from tqdm import tqdm
from keras.models import load_model, Model
from scipy.stats import gaussian_kde
import sympy


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dataset, dtype, layer_names):
    """Determine saved path of ats and pred

    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.

    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + dtype + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + dtype + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10,
        num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")
        pred = model.predict_classes(dataset, batch_size=batch_size, verbose=1)
        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

        print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.

    Args:
        at (list): List of activation traces of an input.        
        train_ats (list): List of activation traces in training set (filtered)
        
    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names, args):
    """Extract ats of train and target inputs. If there are saved files, then skip it.

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """

    saved_train_path = _get_saved_path(args.save_path, args.d, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))

    saved_target_path = _get_saved_path(
        args.save_path, args.d, target_name, layer_names
    )
    if os.path.exists(saved_target_path[0]):
        print(infog("Found saved {} ATs, skip serving").format(target_name))
        # In case target_ats is stored in a disk
        target_ats = np.load(saved_target_path[0])
        target_pred = np.load(saved_target_path[1])
    else:
        target_ats, target_pred = get_ats(
            model,
            x_target,
            target_name,
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_target_path,
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))

    return train_ats, train_pred, target_ats, target_pred


def fetch_dsa(model, x_train, x_target, target_name, layer_names, args):
    """Distance-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard args.

    Returns:
        dsa (list): List of dsa for each target input.
    """

    assert args.is_classification == True

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa = []

    print(prefix + "Fetching DSA")
    for i, at in enumerate(tqdm(target_ats)):
        label = target_pred[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist)

    return dsa


def to_positive_definitive(M):
    M = (M + M.T) * 0.5
    k = 1
    I = np.eye(M.shape[0])
    w, v = np.linalg.eig(M)
    min_eig = v.min()
    M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M


def validate_positive_definitive(M):
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        M = to_positive_definitive(M)
    # Print the eigenvalues of the Matrix
    print(np.linalg.eigvalsh(p))
    return M


class GaussianKde(gaussian_kde):
    """
    Drop-in replacement for gaussian_kde that adds the class attribute EPSILON
    to the covmat eigenvalues, to prevent exceptions due to numerical error.
    """

    EPSILON = 1e-10  # adjust this at will

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                         bias=False,
                                                         aweights=self.weights))
            self._data_inv_cov = np.linalg.pinv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor ** 2
        self.inv_cov = self._data_inv_cov / self.factor ** 2
        # self.inv_cov = validate_positive_definitive(self.inv_cov)
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance))


def _get_kdes(train_ats, train_pred, class_matrix, is_classification=True, num_classes=10):
    """Kernel density estimation

    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard args.

    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    var_threshold = 1e-5

    removed_cols = []
    if is_classification:
        for label in range(num_classes):
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            for i in range(col_vectors.shape[0]):
                if (
                        np.var(col_vectors[i]) < var_threshold
                        and i not in removed_cols
                ):
                    removed_cols.append(i)
        kdes = {}
        for label in tqdm(range(num_classes), desc="kde"):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)

            if refined_ats.shape[0] == 0:
                print("ats were removed by threshold {}".format(var_threshold))
                break
            # try:
            # refined_ats = to_positive_definitive(refined_ats)
            indexes = sympy.Matrix(refined_ats).T.rref()
            kdes[label] = GaussianKde(refined_ats)
            # except:

    else:
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < var_threshold:
                removed_cols.append(i)

        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print("ats were removed by threshold {}".format(var_threshold))
        kdes = [gaussian_kde(refined_ats)]

    print("The number of removed columns: {}".format(len(removed_cols)))

    return kdes, removed_cols


def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    # if np.linalg.det(refined_at) == 0:
    #     refined_at = np.linalg.inv(np.linalg.pinv(refined_at))
    # sa = np.asscalar(-kde.logpdf(np.transpose(refined_at)))
    try:
        sa = np.asscalar(-np.log(kde.pdf(np.transpose(refined_at))))
    except:
        sa = np.nan
    # sa21 = np.asscalar(-kde[1].logpdf(np.transpose(refined_at)))
    # sa22 = np.asscalar(-np.log(kde[1].pdf(np.transpose(refined_at))))
    # if sa == np.nan or sa == np.inf or sa == -np.inf:
    #     a = 0
    return sa


def fetch_lsa(model, x_train, x_target, target_name, layer_names, args):
    """Likelihood-based SA

    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or[] adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: Keyboard args.

    Returns:
        lsa (list): List of lsa for each target input.
    """

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    if args.is_classification:
        for i, label in enumerate(train_pred):
            if label not in class_matrix:
                class_matrix[label] = []
            class_matrix[label].append(i)

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args)

    lsa = []
    print(prefix + "Fetching LSA")
    if args.is_classification:
        for i, at in enumerate(tqdm(target_ats)):
            label = target_pred[i]
            kde = kdes[label]
            lsa.append(_get_lsa(kde, at, removed_cols))
    else:
        kde = kdes[0]
        for at in tqdm(target_ats):
            lsa.append(_get_lsa(kde, at, removed_cols))

    return lsa


def get_sc(lower, upper, k, sa):
    """Surprise Coverage

    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.

    Returns:
        cov (int): Surprise coverage.
    """

    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    # return len(list(set(buckets))) / float(k) * 100
    return buckets