from __future__ import division
from scitbx.math import dihedral_angle
from mmtbx.ligands.ready_set_utils import construct_xyz
from mmtbx.ligands.ready_set_utils import generate_atom_group_atom_names
from mmtbx.ligands.ready_set_utils import new_atom_with_inheritance

def add_side_chain_acid_hydrogens_to_atom_group(atom_group,
                                                anchors=None,
                                                configuration_index=0,
                                                bond_length=0.95,
                                                ):
  """Add hydrogen atoms to side-chain acid in place

  Args:
      atom_group (TYPE): Atom group
      anchors (None, optional): Atoms that specify the acids moeity
      configuration_index (int, optional): Configuration to return

  """
  c, o1, o2 = anchors
  if configuration_index>=2:
    tmp = o1.name
    o1.name = o2.name
    o2.name = tmp
    tmp = o1
    o1 = o2
    o2 = tmp
    configuration_index=configuration_index%2
  if o2.name==' OD2':
    name = ' HD2'
    atom = atom_group.get_atom('CB')
  elif o2.name==' OE2':
    name = ' HE2'
    atom = atom_group.get_atom('CG')
  else: assert 0
  element='H'
  dihedral = dihedral_angle(sites=[atom.xyz,
                                   c.xyz,
                                   o1.xyz,
                                   o2.xyz,
                                 ],
                            deg=True)
  ro2 = construct_xyz(o2, bond_length,
                      c, 120.,
                      o1, dihedral,
                      period=2,
                     )
  i = configuration_index
  atom = atom_group.get_atom(name.strip())
  if atom:
    pass #atom.xyz = ro2[i]
  else:
    atom = new_atom_with_inheritance(name, element, ro2[i], o2)
    atom_group.append_atom(atom)

def add_side_chain_acid_hydrogens_to_residue_group(residue_group,
                                                   configuration_index=0,
                                                   ):
  """Adds hydrogen atoms to side-chain acid.

  Args:
      residue_group (TYPE): Specific residue group
  """
  def _get_atom_names(residue_group):
    assert len(residue_group.atom_groups())==1
    atom_group = residue_group.atom_groups()[0]
    lookup = {'ASP' : ['CG', 'OD1', 'OD2'],
              'GLU' : ['CD', 'OE1', 'OE2'],
    }
    return lookup.get(atom_group.resname, [])
  #
  atoms = _get_atom_names(residue_group)
  for ag, atoms in generate_atom_group_atom_names(residue_group,
                                                  atoms,
                                                  ):
    if ag is None: continue
    tmp = add_side_chain_acid_hydrogens_to_atom_group(
      ag,
      # append_to_end_of_model=append_to_end_of_model,
      anchors = atoms,
      configuration_index=configuration_index,
    )

def add_side_chain_acid_hydrogens(hierarchy,
                                  configuration_index=0,
                                  ):
  """Add hydrogen atoms to every side-chain acid (ASP and GLU). Not very
  useful as adding to a single residue group (below) would be more prectical.

  Args:
      hierarchy (TYPE): Model hierarchy
      configuration_index (int, optional): Defaults to zero. Determines which
        of the four configurations the added hydrogen will be:
          0 - Current Ox2 gets Hx2 (x=D,E) pointing out
          1 - Current Ox2 gets Hx2 (x=D,E) pointing in
          2 - Current Ox1 gets swapped with Ox2, gets Hx2 (x=D,E) pointing out
          3 - Current Ox1 gets swapped with Ox2, gets Hx2 (x=D,E) pointing in
  """
  for rg in hierarchy.residue_groups():
    for ag in rg.atom_groups():
      if ag.resname in ['ASP', 'GLU']:
        add_side_chain_acid_hydrogens_to_residue_group(
          rg,
          configuration_index=configuration_index,
          )
