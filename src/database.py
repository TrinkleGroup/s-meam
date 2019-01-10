import os
import glob
import pickle
import numpy as np
from collections import namedtuple
import src.lammpsTools

# allowed 'types': 'energy' or 'forces'
entry = namedtuple(
    'entry', 'struct_name value natoms type ref_struct'
)

class Database:
    def __init__(self, structure_folder_name="", info_folder_name=""):

        self.structures_folder_path = os.path.abspath(structure_folder_name)
        self.true_values_folder_path = os.path.abspath(info_folder_name)

        self.structures_metadata = {}
        self.true_values_metadata = {}

        self.unique_structs = []
        self.unique_natoms = []
        self.entries = []

        # self.natoms = {}
        # self.true_energies = {}
        # self.true_forces = {}
        self.force_weighting = {}
        # self.reference_structs = {}
        # self.reference_energy = None
        # 
        # self.weights = {}

        # if structure_folder_name != "":
        #     self.load_structures()

        # if info_folder_name != "":
        #     self.load_true_values()

    @classmethod
    def manual_init(cls, structures, energies, forces, weights, ref_struct,
                    ref_eng):

        new_db = Database()

        new_db.structures = structures
        new_db.true_energies = energies
        new_db.true_forces = forces
        new_db.weights = weights
        new_db.reference_struct = ref_struct
        new_db.reference_energy = ref_eng

        return new_db

    def load_structures(self):
        sorted_names = sorted(glob.glob(self.structures_folder_path + "/*"))
        for file_name in sorted_names:
            if "metadata" not in file_name:
                f = open(file_name, 'rb')
                struct = pickle.load(f)
                f.close()

                # assumes file_name is of form "*/[struct_name].pkl"
                short_name = os.path.split(file_name)[-1]
                short_name = os.path.splitext(short_name)[0]

                self.structures[short_name] = struct
                self.weights[short_name] = 1
            else:
                # each line should be of the form "[data_name] [data]"
                for line in open(file_name):
                    line = line.strip()
                    tag = line.split(" ")[0]
                    info = line.split(" ")[1:]
                    self.structures_metadata[tag] = info

    def load_true_values(self):
        min_energy = None

        # for short_name in self.structures.keys():
        for file_name in glob.glob(self.true_values_folder_path + "/*"):
            # file_name = self.true_values_folder_path + "/info." + short_name
            if "metadata" not in file_name:

                # assumes file_name is of form "*/info.[struct_name]"
                short_name = os.path.split(file_name)[-1]
                short_name = '.'.join(short_name.split('.')[1:])

                # assumes the first line of the file is the energy
                eng = np.genfromtxt(file_name, max_rows=1)

                if (min_energy is None) or (eng < min_energy):
                    min_energy = eng
                    self.reference_struct = short_name
                    self.reference_energy = min_energy

                # assumes the next N lines are the forces on the N atoms
                fcs = np.genfromtxt(file_name, skip_header=1)

                self.true_energies[short_name] = eng
                self.true_forces[short_name] = fcs
                self.natoms[short_name] = fcs.shape[0]
            else:
                # each line should be of the form "[data_name] [data]"
                for line in open(file_name):
                    line = line.strip()
                    tag = line.split(" ")[0]
                    info = line.split(" ")[1:]
                    self.true_values_metadata[tag] = info

    def print_metadata(self):
        print("Reference structure:", self.reference_struct)
        print("Metadata (structures):")
        for tag, info in self.structures_metadata.items():
            print(tag + ":", " ".join(info))

        print("Metadata (true values):")
        for tag, info in self.true_values_metadata.items():
            print(tag + ":", " ".join(info))

    def __len__(self):
        return len(self.natoms)

    def read_pinchao_formatting(self, directory_path):
        # for file_name in glob.glob(os.path.join(directory_path, 'force_*')):
        #     with open(file_name) as f:
        #         struct_name = f.readline().strip()

        #     full = np.genfromtxt(file_name, skip_header=1)

        #     self.true_forces[struct_name] = full[:, 1:] * full[:, 0, np.newaxis]
        #     self.force_weighting[struct_name] = full[:, 0]
        #     self.natoms[struct_name] = full.shape[0]

        already_added = []

        with open(os.path.join(directory_path, 'FittingDataEnergy.dat')) as f:

            for _ in range(3):
                _ = f.readline() # remove headers

            for line in f:
                struct_name, natoms, _, ref_name, weight = line.split(" ")
                natoms = int(natoms)
                ref_name = ref_name.strip()

                eng = float(f.readline().strip())
                weight = float(weight)

                # 'entry', 'struct_name value natoms type ref_struct'
                check_entry = entry(struct_name, None, None, 'energy', ref_name)

                if check_entry not in already_added:
                    already_added.append(check_entry)
                    self.entries.append(
                        entry(struct_name, eng, natoms, 'energy', ref_name)
                    )

                if struct_name not in self.unique_structs:
                    self.unique_structs.append(struct_name)
                    self.unique_natoms.append(natoms)

                ref_atoms_object = src.lammpsTools.atoms_from_file(
                    os.path.join(directory_path, ref_name), ['H', 'He']
                )

                ref_natoms = len(ref_atoms_object)

                if ref_name not in self.unique_structs:
                    self.unique_structs.append(ref_name)
                    self.unique_natoms.append(ref_natoms)


                file_name = os.path.join(directory_path, 'force_' + struct_name)
                full = np.genfromtxt(file_name, skip_header=1)

                struct_forces = full[:, 1:] * full[:, 0, np.newaxis]
                self.force_weighting[struct_name] = full[:, 0]

                file_name = os.path.join(directory_path, 'force_' + ref_name)
                full = np.genfromtxt(file_name, skip_header=1)

                ref_forces = full[:, 1:] * full[:, 0, np.newaxis]
                self.force_weighting[ref_name] = full[:, 0]


                # 'entry', 'struct_name value natoms type ref_struct'


                check_entry = entry(struct_name, None, None, 'forces', None)

                if check_entry not in already_added:
                    already_added.append(check_entry)

                    self.entries.append(
                        entry(
                            struct_name, struct_forces, natoms, 'forces', None
                        )
                    )

                check_entry = entry(ref_name, None, None, 'forces', None)

                if check_entry not in already_added:
                    already_added.append(check_entry)

                    self.entries.append(
                        entry(
                            ref_name, ref_forces, ref_natoms, 'forces', None
                        )
                    )


if __name__ == "__main__":
    db = Database(
        "data/fitting_databases/fixU-clean/structures",
        "data/fitting_databases/fixU-clean/rhophi/info",
        )

    print("Database length:", len(db))
    print("Keys:", db.true_forces.keys())
    print()
    db.print_metadata()
