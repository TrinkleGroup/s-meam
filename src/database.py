import os
import glob
import pickle
import numpy as np
import src.lammpsTools

class Database:
    def __init__(self, structure_folder_name, info_folder_name, atom_types):

        self.structures_folder_path = os.path.abspath(structure_folder_name)
        self.true_values_folder_path = os.path.abspath(info_folder_name)

        self.structures_meta_data = {}
        self.true_values_meta_data = {}

        self.structures = {}
        self.true_energies = {}
        self.true_forces = {}

        self.load_structures()
        self.load_true_values()

    def load_structures(self):
        for file_name in glob.glob(self.structures_folder_path + "/*"):
            if "meta_data" not in file_name:
                struct = pickle.load(open(file_name, 'rb'))

                # assumes file_name is of form "*/[struct_name].pkl"
                short_name = os.path.split(file_name)[-1]
                short_name = os.path.splitext(short_name)[0]

                self.structures[short_name] = struct
            else:
                # each line should be of the form "[data_name] [data]"
                for line in open(file_name):
                    line = line.strip()
                    tag = line.split(" ")[0]
                    info = line.split(" ")[1:]
                    self.structures_meta_data[tag] = info

    def load_true_values(self):
        for file_name in glob.glob(self.true_values_folder_path + "/*"):
            if "meta_data" not in file_name:

                # assumes file_name is of form "*/info.[struct_name]"
                short_name = os.path.split(file_name)[-1]
                short_name = os.path.splitext(short_name)[-1][1:]

                # assumes the first line of the file is the energy
                eng = np.genfromtxt(open(file_name, 'rb'), max_rows=1)

                # assumes the next N lines are the forces on the N atoms
                fcs = np.genfromtxt(open(file_name, 'rb'), skip_header=1)

                self.true_energies[short_name] = eng
                self.true_energies[short_name] = eng
            else:
                # each line should be of the form "[data_name] [data]"
                for line in open(file_name):
                    line = line.strip()
                    tag = line.split(" ")[0]
                    info = line.split(" ")[1:]
                    self.true_values_meta_data[tag] = info

    def print_meta_data(self):
        print("Structures meta data")
        for tag, info in self.structures_meta_data.items():
            print(tag + ":", " ".join(info))

        print()
        print("True values meta data")
        for tag, info in self.true_values_meta_data.items():
            print(tag + ":", " ".join(info))

if __name__ == "__main__":
    db = Database(
        "data/fitting_databases/fixU-clean/structures",
        "data/fitting_databases/fixU-clean/rhophi/info",
        ['H','He']
        )

    db.print_meta_data()
