#import os
#
#forces_files = grep('../lammps-versions/*_forces.dat')
#
#for f in forces_files:
#    data = np.genfromtxt(f, comments="#")
#    infile = open(f, 'r')
#
#    structName = os.path.split(f.readline().strip())[-1]
#    structName,ext = os.path.splitext(structName)
#
#    outfile = open(structName + ext + '.forces_tests.py', 'w')
#
#    natoms = int(f.readline().strip()[-1])
#    
#    line = f.readline() # potential type
#    pnum = 0
#    while line
#        ptype = line.strip()[1:]
#
#        for j in xrange(natoms):
#            line = f.readline() # skip data that we pre-read
#
#        forces = data[pnum*natoms:(pnum+1)*natoms]
#        print(forces)
#
#        outfile.write("class_" + structName + '_' + ext + "_forces(unittest.TestCase):\n\n")
#        outfile.write("\t def test_" + ptype + "_forces(self):\n")
#        outfile.write("\t\\tp = MEAM(\"./test-files/TiO." + ptype + "spline\")\n")
#        outfile.write("\t\tatoms = lammpsTools.atoms_from_file(\"./test-files/" + structName + '.' + ext + "\", ['Ti','O'])\n\n")
#
#        outfile.write("\t\tfor i in xrange(" + str(natoms) + "):\n")
#        outfile.write("\t\t\tnp.testing.assert_allclose()")
#
#    infile.close()
#    outfile.close()

import os
import glob

tempfile = open('tmp.py', 'r')
lines = ''.join(tempfile.readlines())
print(lines)

outfile = open("force_tests.py", 'w')
outfile.write("class forces_all_structs(unittest.TestCase):\n")
outfile.write("    \"\"\" Test force calculations for all structures in Zhang/Trinkle Ti-O database\"\"\"\n\n")

allFiles = glob.glob("./all-structs/*")
print(len(allFiles))
for f in allFiles:
    structName1 = os.path.split(f)[-1]
    structName,ext = os.path.splitext(structName1)
    
    outfile.write("    def test_" + '_'.join(structName1.split('.')) + "_forces(self):\n")
    outfile.write("        dataName = \"" + structName + ext + "\"\n\n")

    outfile.write(lines)
    outfile.write("\n")
