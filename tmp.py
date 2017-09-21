        ptype = "meam"
        p = MEAM("./test-files/TiO." + ptype + ".spline")
        atoms = lammpsTools.atoms_from_file("./test-files/data."+dataName,\
                ['Ti','O'])

        data = np.genfromtxt("./test-files/" + dataName + "_" + ptype +\
                "_forces.dat")
        val = p.compute_forces(atoms)

        np.testing.assert_allclose(val, data, atol=1e-6)
