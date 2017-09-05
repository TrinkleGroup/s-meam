        nl = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=True, skin=0.0)
        nl_noboth = NeighborList(np.ones(len(atoms))*(self.cutoff/2),\
                self_interaction=False, bothways=False, skin=0.0)
        nl.build(atoms)
        nl_noboth.build(atoms)

        plot_three = []
        ucounts = 0
        for i in xrange(natoms):
            itype = symbol_to_type(atoms[i].symbol, self.types)

            neighbors = nl.get_neighbors(i)[0]
            neighbors_noboth = nl_noboth.get_neighbors(i)[0]

            pairs = itertools.product([i], neighbors)
            pairs_noboth = itertools.product([i], neighbors_noboth)
            neighbors_without_j = neighbors
            #print(type(neighbors_without_j))

            # TODO: workaround for this if branch
            if len(neighbors) > 0:
                tripcounter = 0
                total_phi = 0.0
                total_u = 0.0
                total_rho = 0.0
                total_ni = 0.0

                u = self.us[i_to_potl(itype)]

                # Calculate pair interactions (phi)
                for pair in pairs_noboth:
                    _,j = pair
                    r_ij = r[i][j]
                    jtype = symbol_to_type(atoms[j].symbol, self.types)

                    phi = self.phis[ij_to_potl(itype,jtype,self.ntypes)]

                    total_phi += phi(r_ij)
                # end phi loop

                # Calculate embedding term (rho)
                #for pair in pairs:
                #    _,j = pair
                #    r_ij= r[i][j]
                #    jtype = symbol_to_type(atoms[j].symbol, self.types)

                #    rho = self.rhos[i_to_potl(jtype)]

                #    total_rho += rho(r_ij)
                ## end rho loop
            
                ## Re-instantiate since this is a generator
                #pairs_noboth = itertools.product([i], neighbors)
                    
                # TODO: problem; times that u(ni_val) is calculated should be
                # once for every atom, INDEPENDENT of # pairs
                # Calculate threebody interactions (u)
                for pair in pairs:

                    _,j = pair
                    r_ij = r[i][j]
                    jtype = symbol_to_type(atoms[j].symbol, self.types)

                    rho = self.rhos[i_to_potl(jtype)]

                    total_ni += rho(r_ij)

                    # Iteratively kill atoms; avoid 2x counting triplets
                    neighbors_without_j = np.delete(neighbors_without_j,\
                            np.where(neighbors_without_j==j))

                    # TODO: only delete after calculation, so that j==k is
                    # included?
                    triplets = itertools.product(pair,neighbors_without_j)

                    for _,k in triplets:
                        r_ik = r[i][k]
                        ktype = symbol_to_type(atoms[k].symbol, self.types)

                        fj = self.fs[i_to_potl(jtype)]
                        fk = self.fs[i_to_potl(ktype)]
                        g = self.gs[ij_to_potl(jtype,ktype,self.ntypes)]

                        a = atoms[j].position-atoms[i].position
                        b = atoms[k].position-atoms[i].position

                        na = np.linalg.norm(a)
                        nb = np.linalg.norm(b)

                        # TODO: try get_dihedral() for angles
                        cos_theta = np.dot(a,b)/na/nb

                        fj_val = fj(r_ij)
                        fk_val = fk(r_ik)
                        g_val = g(r_ik)

                        total_ni += fj_val*fk_val*g_val
                        tripcounter += 1
                    # end triplet loop

                    #plot_three.append(total_threebody)
                    #ni_val = total_rho + total_threebody
                    #u_val = u(ni_val)

                    #total_u += u_val
                # end u loop

                print("ni_val = %f || " % total_ni),
                print("%f" % u(total_ni))
                #print("zero_atom_energy = %f" % self.zero_atom_energies[i_to_potl(itype)])
                ucounts += 1
                total_pe += total_phi + u(total_ni) -\
                        self.zero_atom_energies[i_to_potl(itype)]


