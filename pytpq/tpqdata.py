import re
import h5py


"""
    Class to handle all the TPQ data that exists for a given qn_label inside the specified directory
    where hdf5 file names must match regex_str (that contains n_iter and n_samples as fixed numbers).
"""
class TPQData:

    """ Constructor of TPQData class """
    def __init__(self, qn_label, directory, regex_str):
        # retrieve the data for a given qn_label (dict mapping seeds to tuples of alpha and beta vectors)
        self.seed_data_dict = read_data(qn_label, directory, regex_str)
        self.qn_label = qn_label
        self.seeds = list(self.seed_data_dict.keys())

        if len(self.seeds) == 0:
            raise ValueError("No seeds to initialize TPQData")
        
        # check if dimensions of alpha, beta (<= n_iter) are the same for all initial vectors
        prev_dim = None
        for alpha_beta_dict in self.seed_data_dict.values():
            alpha_dim = len(alpha_beta_dict["alpha"])
            if prev_dim is None:
                prev_dim = alpha_dim
            elif prev_dim != alpha_dim:
                raise ValueError("Different dimensions of alpha encountered: {} and {}".format(prev_dim, alpha_dim))
        self.dim_alpha = prev_dim

    """ Return alpha and beta for a given seed """        
    def dataset(self, seed):
        data = self.seed_data_dict[seed]
        alpha = data[0]
        beta = data[1]
        return alpha, beta
    


"""

    ---------- Functions used for reading in TPQ data ---------- 

"""



"""
    Find all HDF5 files contained in "directory" and all its subdirectories
    and return a dictionary with keys being the qns_labels and values being a
    vector of paths to TPQ files with these quantum numbers.

    - base_path = path below which the function searches for hdf5 files.
    - regex_str = the pattern used for matching file names of hdf5 files (n_iter and n_samples are fixed therein!)

"""
def find_all_tpq_files(directory, regex_str):
    print("----- Scanning for HDF5 files in: {} -----".format(directory))
    # list of all hdf5 files in all subdirectories of base_path
    hdf5_files = list(directory.rglob("*.h5"))
    print("Found {} hdf5 files.".format(len(hdf5_files)))
    # dictionary mapping qns_labels to list of files
    qns_to_path_dict = dict()
    n_files = 0
    for h5path in hdf5_files:
        # try to match file name and extract quantum numbers
        match_res = re.match(regex_str, h5path.name)
        if match_res is not None:
            n_files += 1
            Sztot = int(match_res.group(1))
            irrep = match_res.group(2)
            if (Sztot, irrep) in qns_to_path_dict:
                qns_to_path_dict[(Sztot, irrep)].append(h5path)
            else:
                qns_to_path_dict[(Sztot, irrep)] = [h5path]
    print("Created quantum-number-resolved dictionary of {} h5 files inside {}.".format(n_files, directory))
    return qns_to_path_dict


def _read_single_hdf5_file(h5path):
    """ Read a single hdf5 file and return the data as a dictionary of seeds mapped to alpha, beta tuples """
    data = dict()
    with h5py.File(h5path, 'r') as hf:
        for seed in hf.keys(): # keys of hdf5 file are assumed to be the seeds of TPQ vectors
            alpha = hf[seed]['alpha']
            beta = hf[seed]['beta']
            data[int(seed)] = (alpha, beta)
    return data


"""
    Read all TPQ hdf5 data files available inside "directory" matching the regular expression of file names,
    i.e., whose quantum numbers correspond to qn_label (n_iter, n_samples fixed through regular expression matching).
"""
def read_data(qn_label, directory, regex_str):
    # get dictionary of paths to hdf5 files for all quantum numbers
    qns_to_path_dict = find_all_tpq_files(directory, regex_str)
    if qn_label not in qns_to_path_dict:
        raise ValueError("No hdf5 files found for quantum numbers: {}".format(qn_label))
    qn_paths = qns_to_path_dict[qn_label]
    seed_data_dict = dict()
    # read all hdf5 files with required quantum numbers in serial
    for h5path in qn_paths:
        data_dict = _read_single_hdf5_file(h5path)
        for seed, (alpha, beta) in data_dict.items():
            if seed in seed_data_dict:
                raise ValueError("Duplicate seed {} encountered when reading hdf5 files.".format(seed))
            seed_data_dict[seed] = (alpha, beta)
    # return dictionary mapping seeds to (alpha, beta) tuples
    return seed_data_dict
