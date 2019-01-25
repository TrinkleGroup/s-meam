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
    def __init__(self, structure_folder_name="", info_folder_name="",
                 ref_name=None):

        self.ref_name = ref_name
        self.structures_folder_path = os.path.abspath(structure_folder_name)
        self.true_values_folder_path = os.path.abspath(info_folder_name)

        self.structures_metadata = {}
        self.true_values_metadata = {}

        self.unique_structs = []
        self.unique_natoms = []
        self.entries = []

        self.force_weighting = {}

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

    def load_structures(self, max_num_structs=None):
        sorted_names = glob.glob(self.true_values_folder_path + "/*")

        if max_num_structs is None:
            load_num = len(sorted_names)
        else:
            load_num = max_num_structs

        additional_names = np.random.choice(
            sorted_names, load_num, replace=False
        )

        to_add = additional_names.tolist()

        ref_path = os.path.join(
            self.true_values_folder_path, "info." + self.ref_name
        )

        if ref_path not in to_add:
            to_add[np.random.randint(len(to_add))] = ref_path


        # to_add = []
        # with open("/home/jvita/scripts/s-meam/data/results/hyojung/names.txt", "r") as f:
        #     for line in f:
        #         clean = "_".join(line.strip().split("/")[1:])
        #         to_add.append(
        #             os.path.join(self.true_values_folder_path, "info."+ clean)
        #         )

        already_added = []
        for file_name in to_add:
            if "metadata" not in file_name:
                f = open(file_name, 'rb')
                f.close()

                # assumes file_name is of form "*/[struct_name].pkl"
                # file_name = os.path.splitext(file_name)[1][1:]
                struct_name = os.path.split(file_name)[-1]
                struct_name = os.path.splitext(struct_name)[1][1:]

                # 'entry', 'struct_name value natoms type ref_struct'
                check_entry = entry(
                    struct_name, None, None, 'energy', self.ref_name
                )

                eng = np.genfromtxt(file_name, max_rows=1)
                fcs = np.genfromtxt(file_name, skip_header=1)
                natoms = fcs.shape[0]

                if check_entry not in already_added:
                    already_added.append(check_entry)
                    self.entries.append(
                        entry(struct_name, eng, natoms, 'energy', self.ref_name)
                    )

                if struct_name not in self.unique_structs:
                    self.unique_structs.append(struct_name)
                    self.unique_natoms.append(natoms)

                # 'entry', 'struct_name value natoms type ref_struct'
                check_entry = entry(struct_name, None, None, 'forces', None)

                if check_entry not in already_added:
                    already_added.append(check_entry)

                    self.entries.append(
                        entry(struct_name, fcs, natoms, 'forces', None)
                    )


            else:
                # each line should be of the form "[data_name] [data]"
                for line in open(file_name):
                    line = line.strip()
                    tag = line.split(" ")[0]
                    info = line.split(" ")[1:]
                    self.structures_metadata[tag] = info

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

    def read_pinchao_formatting(self, directory_path, db_type):

        if db_type == 'fitting':
            load_file = 'FittingDataEnergy.dat'
        elif db_type == 'testing':
            load_file = 'TestingDataEnergy.dat'
        else:
            raise ValueError(
                "Invalid database type: must be 'fitting' or 'testing'"
            )

        already_added = []

        with open(os.path.join(directory_path, load_file)) as f:

            for _ in range(3):
                _ = f.readline() # remove headers

            for line in f:
                if db_type == 'fitting':
                    struct_name, natoms, _, ref_name, weight = line.split(" ")
                else:
                    struct_name, natoms, _, ref_name = line.split(" ")
                    weight = 1

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
