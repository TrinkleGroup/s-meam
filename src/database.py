import os
import glob
import pickle
import numpy as np
import src.lammpsTools

class Database:
    def __init__(self, structure_folder_name="", info_folder_name=""):

        self.structures_folder_path = os.path.abspath(structure_folder_name)
        self.true_values_folder_path = os.path.abspath(info_folder_name)

        self.structures_metadata = {}
        self.true_values_metadata = {}

        self.structures = {}
        self.true_energies = {}
        self.true_forces = {}
        self.reference_struct = None
        self.reference_energy = None

        self.weights = {}

        if structure_folder_name != "":
            self.load_structures()

        if info_folder_name != "":
            self.load_true_values()

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
        for file_name in glob.glob(self.structures_folder_path + "/*"):
            if "metadata" not in file_name:
                struct = pickle.load(open(file_name, 'rb'))

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

        for file_name in glob.glob(self.true_values_folder_path + "/*"):
            if "metadata" not in file_name:

                # assumes file_name is of form "*/info.[struct_name]"
                short_name = os.path.split(file_name)[-1]
                short_name = '.'.join(short_name.split('.')[1:])

                # assumes the first line of the file is the energy
                eng = np.genfromtxt(open(file_name, 'rb'), max_rows=1)

                if (min_energy is None) or (eng < min_energy):
                    min_energy = eng
                    self.reference_struct = short_name
                    self.reference_energy = min_energy

                # assumes the next N lines are the forces on the N atoms
                fcs = np.genfromtxt(open(file_name, 'rb'), skip_header=1)

                self.true_energies[short_name] = eng
                self.true_forces[short_name] = fcs
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

        print()
        print("Metadata (true values):")
        for tag, info in self.true_values_metadata.items():
            print(tag + ":", " ".join(info))
        print(flush=True)

    def __len__(self):
        return len(self.structures)

if __name__ == "__main__":
    db = Database(
        "data/fitting_databases/fixU-clean/structures",
        "data/fitting_databases/fixU-clean/rhophi/info",
        ['H','He']
        )

    print("Database length:", len(db))
    print("Keys:", db.true_forces.keys())
    print()
    db.print_metadata()
