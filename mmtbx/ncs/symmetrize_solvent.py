from __future__ import absolute_import, division, print_function
import iotbx
from iotbx import phil
from iotbx.pdb.utils import all_chain_ids
from iotbx.ncs import input as ncs_input
from iotbx.ncs import ncs_search_options, ncs_group_phil_str
import mmtbx
from mmtbx.ncs.ncs_search import get_chains_info
from libtbx.test_utils import approx_equal
from scitbx.matrix import rec
from cctbx.array_family import flex
from cctbx.maptbx.box import shift_and_box_model

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  from sklearn.neighbors import KDTree

import os
import numpy as np

__doc__ = """
Functionality here aims to deal with a model that should obey strict
NCS because it originates from a cryo-EM map, but has become asymmetric
due to solvent picking.

There are multiple ways to handle the new solvent identifiers (serial, resid,
chain id)

Method 1: solvent_chain_mode="unique"
The simplest, add NEW solvent to a new unique chain id. Number resid from 1,
and take the next available atom serial number from the reference model.

Method 2: solvent_chain_mode="nearest"
Assign NEW solvent to the nearest macromolecule chain, increment resid above
the highest existing resid.


Usage:   symmetrizer = SymmetrizeSolvent(model)
         symmetrized_model = symmetrizer.run()

"""

# TODO: Improve the control over solvent residue numbering.
#       A fixed increment above the protein would be useful.
# TODO: Currently the expanded model is generated by applying ncs to the
#       reference. This means that differences in the copy protein will be lost.
#       Need to discuss what is the best behavior.
# TODO: Add checks that the transformations are similar when merging ncs groups
# TODO: More complex tests. Need to find a symmetric map to test
# TODO: Need a more comprehensive ion selection string
# TODO: Better organize order of solvent. ie, all HOH after ions


ions = ["Na", "Cl", "Fe", "K", "Mn", "Mg", "Zn", "Ca", "Cu", "Co", "Ni", "Cd"]
solvent_sel_str = "water or " + " or ".join(
  ["resname %s" % e.upper() for e in ions]
)


class SymmetrizeSolvent:
  """
  Initialize with a model an optionally a map. During the init phase, the ncs
  groups are set up. If a map is provided, ncs transformations are taken
  from the map.

  Calling the run() method will return a symmeterized model. Note that this
  involves expanding the reference, so any protein differences in the copies
  will not be retained.

  """

  def __init__(
    self,
    model,
    mmm=None,
    take_ncs_from_map=True,
    solvent_sel_str=solvent_sel_str,
    solvent_chain_mode="nearest",
    write_debug=False,
    write_debug_dir=os.getcwd(),
    out=None,
  ):
    if model.crystal_symmetry() is None:
      model = shift_and_box_model(model, shift_model=False)
    self.model = model
    self.mmm = mmm
    self.solvent_sel_str = solvent_sel_str
    assert solvent_chain_mode in ["nearest", "unique"]
    self.solvent_chain_mode = solvent_chain_mode
    self.write_debug = write_debug
    self.write_debug_dir = write_debug_dir
    self.out = out
    self.devnull = open(os.devnull, "w")

    # set up ncs relationships
    self.ncs_input_obj, self.ncs_group = self.generate_ncs_group()

    if self.assert_ncs_covers_all(self.model, self.ncs_group) is False:
      print(
        "NCS selection does not cover all atoms, will try to add solvent "
        "to the nearest non-solvent chain"
      )
      self.model = self.solvent_to_nearest_chain(self.model)
      self.ncs_group = self.generate_ncs_group()
      assert self.assert_ncs_covers_all(self.model, self.ncs_group)

    # modify ncs_group transformations based on map
    if take_ncs_from_map and mmm is None:
      print(
        "\nNo map model manager. Cannot take symmetry from map...",
        file=self.out,
      )
    elif take_ncs_from_map:
      print("\nSearching for symmetry in map...", file=self.out)
      self.ncs_group = self.take_ncs_from_map(self.mmm, self.ncs_group)

  def run(self):
    """
    Symmetrize the solvent.
    1. Extract ref/copies models
    2. Align to reference model
    3. Find copy solvent missing in reference
    4. Add new solvent to reference
    5. Expand the reference (and solvent) to create a new symmetrized model.
    """
    # get all relevant model objects to work with
    self.master_sol_m = self.model.select(
      self.model.selection(self.solvent_sel_str)
    )
    self.master_nosol_m = self.model.select(
      ~self.model.selection(self.solvent_sel_str)
    )

    self.ref_all_m = self.model.select(
      self.model.selection(self.ncs_group.master_str_selection)
    )
    self.ref_sol_m = self.ref_all_m.select(
      self.ref_all_m.selection(self.solvent_sel_str)
    )
    self.ref_nosol_m = self.ref_all_m.select(
      ~self.ref_all_m.selection(self.solvent_sel_str)
    )
    self.copies_all_m = [
      self.model.select(self.model.selection(ncs_copy.str_selection))
      for ncs_copy in self.ncs_group.copies
    ]
    self.copies_sol_m = [
      copy_all_m.select(copy_all_m.selection(self.solvent_sel_str))
      for copy_all_m in self.copies_all_m
    ]
    self.copies_all_m_collapse = self.collapse_ncs_copies(
      self.copies_all_m, self.ncs_group, inplace=False
    )

    self.copies_sol_m_collapse = self.collapse_ncs_copies(
      self.copies_sol_m, self.ncs_group, inplace=False
    )

    # optionally write collapsed models for inspection
    if self.write_debug:
      print(
        self.ref_all_m.model_as_pdb(),
        file=open(
          os.path.join(self.write_debug_dir, "ncs_collapsed_00_reference.pdb"),
          "w",
        ),
      )
      for i, copy_model in enumerate(self.copies_all_m_collapse):
        print(
          copy_model.model_as_pdb(),
          file=open(
            os.path.join(
              self.write_debug_dir,
              "ncs_collapsed_" + str(i + 1).zfill(2) + "_copy.pdb",
            ),
            "w",
          ),
        )

    # get the reference solvent that contains additions from th) copies
    self.new_sol = self.new_solvent_for_ref(
      self.ref_sol_m, self.copies_sol_m_collapse
    )

    # method 1
    # composed_ref = self.compose_models([self.ref_all_m,self.new_sol])
    self.ref_all_sym = self.solvent_to_nearest_chain(
      self.ref_all_m, solvent_model=self.new_sol
    )

    # expand copies
    self.expand_ref_copies = self.expand_ncs_copies(
      self.ref_all_sym, self.ncs_group
    )

    # rename to original original chain ids
    all_atoms = self.model.get_hierarchy().atoms()
    ref_chain_ids = [
      all_atoms[i].parent().parent().parent().id
      for i in self.ncs_group.master_iselection
    ]
    copy_chain_ids = []
    for ncs_copy in self.ncs_group.copies:
      copy_chain_ids.append(
        [
          all_atoms[i].parent().parent().parent().id
          for i in ncs_copy.iselection
        ]
      )

    chain_match_dict = {}
    for chain_id in ref_chain_ids:
      if chain_id not in chain_match_dict:
        chain_match_dict[chain_id] = []
    for i, copy_chain in enumerate(copy_chain_ids):
      for j, chain_id in enumerate(copy_chain):
        ref_match = ref_chain_ids[j]
        if chain_id not in chain_match_dict[ref_match]:
          chain_match_dict[ref_match].append(chain_id)

    for i, ref_copy in enumerate(self.expand_ref_copies):
      for ref_cid, copy_cid_list in chain_match_dict.items():
        new_cid = copy_cid_list[i]
        ref_copy.get_hierarchy().rename_chain_id(ref_cid, new_cid)

    # compose expanded copies
    symmetrized_model = self.compose_models(
      [self.ref_all_sym] + self.expand_ref_copies
    )

    self._symmetrized_model = symmetrized_model
    self.assert_solvent_symmetric(self._symmetrized_model, self.ncs_group)
    return symmetrized_model

  @property
  def symmetrized_model(self):
    if not hasattr(self, "_symmetrized_model"):
      symmetrized = self.run()  # sets _symmetrized_model
    return self._symmetrized_model

  def generate_ncs_group(self):
    """
    Detect ncs groups using iotbx.ncs.input
    If more than 1 group is detected, the function calls
    force_single_group_ncs to combine them and then re-submits it to
    iotbx.ncs.inpu for validation.

    All selection syntax other than chain id is stripped out. This is done to
    allow for differences between chains (solvent, altlocs, etc)
    """
    ncs_search_params = (
      phil.parse(input_string=ncs_search_options, process_includes=True)
      .extract()
      .ncs_search
    )
    ncs_search_params.exclude_selection = (
      "element H or element D or " "" + self.solvent_sel_str
    )
    ncs_input_obj = ncs_input(
      hierarchy=self.model.get_hierarchy(),
      params=ncs_search_params,
      log=self.out,
    )
    print("Found NCS", file=self.out)
    print(ncs_input_obj.print_ncs_phil_param(), file=self.out)

    if len(ncs_input_obj.ncs_restraints_group_list) == 1:
      print("Single NCS group detected, good.", file=self.out)

    else:
      print(
        "\nMultiple NCS groups detected. Will try to merge into one...\n",
        file=self.out,
      )
      massaged_phil_groups = self.force_single_group_ncs(ncs_input_obj)
      ncs_input_obj = ncs_input(
        hierarchy=self.model.get_hierarchy(),
        ncs_phil_groups=massaged_phil_groups,
        log=self.out,
      )
      assert ncs_input_obj.phil_groups_modified is False
    ncs_group = ncs_input_obj.ncs_restraints_group_list[0]

    return ncs_input_obj, ncs_group

  def force_single_group_ncs(self, ncs_input_obj):
    """
    This function does two things:
    1. Attempts to ignore all other selection syntax in the ncs_input_obj except
       for chain selections.
    2. Attempts to combine multiple NCS groups into a single NCS group

    It then re-submits the new NCS groups to iotbx.ncs.input for validation.
    This works with the models tested, but probably it would be better to
    integrate this functionality with iotbx.ncs.input (simple_ncs_from_pdb)
    """

    def only_chain(sel_str):
      split = sel_str.split()
      chain_work_indices = [
        i for i, word in enumerate(split) if "chain" in word
      ]
      chain_id_indices = [i + 1 for i in chain_work_indices]
      sel_str = ""
      for i, chain_i in enumerate(chain_id_indices):
        chain_id = "'" + split[chain_i].strip("'") + "'"
        sel_str += "chain " + chain_id
        if i < len(chain_id_indices) - 1:
          sel_str += " or "
      return sel_str

    # build a list of lists where each row is one ref/copy and each column is
    # The selection string for a group
    sel_str_components = [
      [] for ncs_group in ncs_input_obj.ncs_restraints_group_list
    ]

    for ncs_group in ncs_input_obj.ncs_restraints_group_list:
      sel_str_components[0].append(ncs_group.master_str_selection)
      for i, ncs_copy in enumerate(ncs_group.copies):
        sel_str_components[i + 1].append(ncs_copy.str_selection)

    # Strip out any selection not related to chain
    sel_str_components_onlychain = []
    for comp_list in sel_str_components:
      only_chain_list = []
      for comp in comp_list:
        only_chain_list.append(only_chain(comp))
      sel_str_components_onlychain.append(only_chain_list)

    # build new user-provided ncs phil groups
    ncs_custom_phil = (
      "ncs_group {\n\treference = "
      "" + " or ".join(sel_str_components_onlychain[0]) + "\n"
    )
    for copy_sel in sel_str_components_onlychain[1:]:
      sel_str = "\tselection = " + " or ".join(copy_sel) + "\n"
      ncs_custom_phil += sel_str
    ncs_custom_phil += "}"
    ncs_search_params = (
      phil.parse(input_string=ncs_search_options, process_includes=True)
      .extract()
      .ncs_search
    )
    ncs_groups_phil = phil.parse(
      input_string=ncs_group_phil_str, process_includes=True
    )
    phil_groups = (
      ncs_groups_phil.fetch(phil.parse(ncs_custom_phil)).extract().ncs_group
    )
    return phil_groups

  def assert_solvent_symmetric(self, model, ncs_group, eps=1e-01):
    """

    Parameters
    ----------
    model : mmtbx.model.model.manager
    ncs_group : mmtbx.ncs.ncs_restraints_group_list.NCS_restraint_group
    eps : the tolerance to consider sites the same
    Returns
    -------
    bool : True if all the solvent in the model is consistent with ncs_group
    """

    ref = model.select(model.selection(ncs_group.master_str_selection))
    ref_sol = ref.select(ref.selection(self.solvent_sel_str))

    copies_m = []
    copies_sol_m = []
    for i, ncs_copy in enumerate(ncs_group.copies):
      copy_m = model.select(model.selection(ncs_copy.str_selection))
      copy_sol_m = copy_m.select(copy_m.selection(self.solvent_sel_str))
      copies_m.append(copy_m)
      copies_sol_m.append(copy_sol_m)

    print("\nVerifying solvent is symmetric:", file=self.out)
    print("Reference n_atoms:", ref_sol.get_number_of_atoms(), file=self.out)
    ref_sol_xyz = ref_sol.get_sites_cart()
    for j, c in enumerate(ncs_group.copies):
      m_copy = copies_sol_m[j]
      print("Copy n_atoms:", m_copy.get_number_of_atoms(), file=self.out)
      ref_tr_sites = c.r.elems * ref_sol_xyz + c.t
      d = flex.sqrt((ref_tr_sites - m_copy.get_sites_cart()).dot())
      assert approx_equal(d.min_max_mean().as_tuple(), [0, 0, 0], eps=eps)
      print("Sites approx_equal using eps", eps, file=self.out)

  def assert_ncs_covers_all(self, model, ncs_group):
    """

    Parameters
    ----------
    model : mmtbx.model.model.manager
    ncs_group : mmtbx.ncs.ncs_restraints_group_list.NCS_restraint_group

    Returns
    -------
    bool : True if the ncs_selections cover all atoms in the model
    """
    success = True
    excluded_sel = flex.bool(model.get_number_of_atoms(), True)
    excluded_sel = excluded_sel.set_selected(
      model.selection(ncs_group.master_str_selection), False
    )
    for ncs_copy in ncs_group.copies:
      excluded_sel = excluded_sel.set_selected(
        model.selection(ncs_copy.str_selection), False
      )
    if excluded_sel.count(True) > 0:
      success = False
    return success

  @staticmethod
  def expand_ncs_copies(ref_model, ncs_group, copy_list=None):
    # collapse reference or reference aligned copies
    if copy_list is None:
      to_expand = [ref_model.deep_copy() for c in ncs_group.copies]
    else:
      to_expand = copy_list
    assert len(to_expand) == len(ncs_group.copies)

    expanded_copies = []
    for i, ncs_copy in enumerate(ncs_group.copies):
      rot_inv = ncs_copy.r.transpose()
      trans_inv = ncs_copy.t
      model = to_expand[i]
      sites = model.get_sites_cart()
      new_sites = rot_inv.elems * sites + trans_inv
      model.set_sites_cart(new_sites)
      expanded_copies.append(model)
    return expanded_copies

  @staticmethod
  def collapse_ncs_copies(copy_list, ncs_group, inplace=True):
    # collapse copies to reference. Model list should include reference
    if inplace:
      to_collapse = copy_list
    else:
      to_collapse = [copy_model.deep_copy() for copy_model in copy_list]

    for ncs_copy, copy_model in zip(ncs_group.copies, to_collapse):
      collapsed_sites = (
        ncs_copy.r.elems * copy_model.get_sites_cart() + ncs_copy.t
      )
      copy_model.set_sites_cart(collapsed_sites)
    return to_collapse

  def solvent_to_nearest_chain(self, model, solvent_model=None):
    """
    Add solvent to nearest macromolecule chain

    Parameters
    ----------
    model : mmtbx.model.model.manager, a model possibly containing solvent
    solvent_model : mmtbx.model.model.manager, a model containing solvent to add

    Returns
    -------
    new_model : mmtbx.model.model.manager, a new model with the solvent added
                to the nearest macromolecule chain
    """
    model = model.deep_copy()
    chain_info = get_chains_info(model.get_hierarchy())

    solvent_selection = model.selection(self.solvent_sel_str)
    non_solvent_model = model.select(~solvent_selection)
    if solvent_model is None:
      solvent_model = model.select(solvent_selection)
      model = non_solvent_model

    solvent_xyz = solvent_model.get_sites_cart().as_numpy_array()
    nonsolvent_xyz = non_solvent_model.get_sites_cart().as_numpy_array()
    non_solvent_chains = [
      atom.parent().parent().parent()
      for atom in non_solvent_model.get_hierarchy().atoms()
    ]
    tree = KDTree(nonsolvent_xyz)
    dists, inds = tree.query(solvent_xyz, k=1)
    dists, inds = dists[:, 0], inds[:, 0]

    atom_serial_i = 1
    atom_serial_max = 0
    for atom in model.get_hierarchy().atoms():
      if atom.serial_as_int() > atom_serial_max:
        atom_serial_max = atom.serial_as_int()

    resseq_current = 1
    for i, atom in enumerate(solvent_model.get_hierarchy().atoms()):
      j = inds[i]  # nearest atom index in non solvent model
      chain = non_solvent_chains[j]

      rg = atom.parent().parent()

      resname = rg.unique_resnames()[0]

      new_ag = iotbx.pdb.hierarchy.atom_group(altloc="", resname=resname)
      for i_seq, new_atom in enumerate(rg.atoms()):
        new_ag.append_atom(atom=new_atom.detached_copy())

      resseq = chain_info[chain.id].resid_max + resseq_current
      resseq_current += 1

      new_rg = iotbx.pdb.hierarchy.residue_group(
        resseq=iotbx.pdb.resseq_encode(value=resseq), icode=" "
      )
      new_rg.append_atom_group(atom_group=new_ag)
      for atom in new_rg.atoms():
        atom.serial = atom_serial_max + atom_serial_i
        atom_serial_i += 1
      new_chain = iotbx.pdb.hierarchy.chain(id=chain.id)
      new_chain.append_residue_group(residue_group=new_rg)
      m = model.get_hierarchy().only_model()
      m.append_chain(new_chain)

    new_model = mmtbx.model.manager(
      model_input=None,
      pdb_hierarchy=model.get_hierarchy(),
      crystal_symmetry=model.crystal_symmetry(),
    )
    new_model.get_hierarchy().atoms().reset_i_seq()

    return new_model

  def new_solvent_for_ref(self, ref_sol_m, copy_sol_m_list, same_cutoff=1.2):
    """
    Take copies that have been aligned with reference, find solvent that is
    missing in the reference, return a model with missing solvent.

    Parameters
    ----------
    ref_sol_m : mmtbx.model.model.manager, the reference model's solvent
    copy_sol_m_list : mmtbx.model.model.manager, the copies' solvent that has
                      been aligned with the reference frame
    same_cutoff : radius in angstrom to consider solvent equivalent after
                  aligning to the reference

    Returns
    -------
    new_sol_m : mmtbx.model.model.manager, the new solvent
    """

    ref_sol_m = ref_sol_m.deep_copy()
    print(
      "Reference solvent atoms:",
      ref_sol_m.get_number_of_atoms(),
      "\n",
      file=self.out,
    )

    new_solvent_rgs = []
    resseq_start = 1
    resseq_current = resseq_start
    for ci, copy_sol_m in enumerate(copy_sol_m_list):

      ref_sol_xyz = ref_sol_m.get_sites_cart().as_numpy_array()

      copy_sol_xyz = copy_sol_m.get_sites_cart().as_numpy_array()
      ref_tree = KDTree(ref_sol_xyz)

      inds, dists = ref_tree.query_radius(
        copy_sol_xyz, r=same_cutoff, return_distance=True
      )
      missing_in_ref = [
        i for i, ind in enumerate(inds) if len(ind) < 1
      ]  # query indices missing in reference

      if len(missing_in_ref) > 0:

        # determine which atom serial number to start with
        copy_atoms = copy_sol_m.get_hierarchy().atoms()
        atom_serial_i = 1
        atom_serial_max = 0
        for atom in ref_sol_m.get_hierarchy().atoms():
          if atom.serial_as_int() > atom_serial_max:
            atom_serial_max = atom.serial_as_int()

        # add missing atoms to reference
        for i, idx in enumerate(missing_in_ref):

          atom = copy_atoms[idx]
          resname = atom.parent().parent().unique_resnames()[0]
          print(
            "Found solvent to add to reference:",
            resname,
            file=self.out,
          )
          new_ag = iotbx.pdb.hierarchy.atom_group(altloc="", resname=resname)
          atom = atom.detached_copy()
          # print("DEBUG: ATOM OLD SERIAL:", atom.serial)
          atom.serial = atom_serial_max + atom_serial_i
          # print("DEBUG: ATOM NEW SERIAL:", atom.serial)
          atom_serial_i += 1
          new_ag.append_atom(atom)

          resseq = resseq_current
          resseq_current += 1

          new_rg = iotbx.pdb.hierarchy.residue_group(
            resseq=iotbx.pdb.resseq_encode(value=resseq), icode=" "
          )
          new_rg.append_atom_group(atom_group=new_ag)
          new_solvent_rgs.append(new_rg.detached_copy())
          chain_id = "Z"  # tmp
          new_solvent_chain = iotbx.pdb.hierarchy.chain(id=chain_id)
          new_solvent_chain.append_residue_group(new_rg)
          model = ref_sol_m.get_hierarchy().only_model()
          model.append_chain(new_solvent_chain)

        ref_sol_m = mmtbx.model.manager(
          model_input=None,
          pdb_hierarchy=ref_sol_m.get_hierarchy(),
          crystal_symmetry=self.model.crystal_symmetry(),
        )
        ref_sol_m.get_hierarchy().atoms().reset_i_seq()

    ref_sol_m.get_hierarchy().atoms().reset_i_seq()
    # build new model with only new solvent
    h = iotbx.pdb.hierarchy.root()
    model = iotbx.pdb.hierarchy.model()
    h.append_model(model)
    chain = iotbx.pdb.hierarchy.chain(id="Z")
    model.append_chain(chain)
    for rg in new_solvent_rgs:
      chain.append_residue_group(rg)
    new_sol_m = mmtbx.model.manager(
      model_input=None,
      pdb_hierarchy=h,
      crystal_symmetry=self.model.crystal_symmetry(),
    )
    new_sol_m.get_hierarchy().atoms().reset_i_seq()
    print(
      "Total new solvent for reference:",
      new_sol_m.get_number_of_atoms(),
      file=self.out,
    )

    return new_sol_m

  # def match_copies(self,model_list1, model_list2):
  #   # returns a list of tuples matching a model
  #   # from the first list to its neighbor in the
  #   # second list.
  #
  #   list2_index = np.concatenate(
  #     [np.full(model.get_number_of_atoms(), i, dtype=int) for
  #      i, model in enumerate(model_list2)])
  #   list2_xyz = np.vstack(
  #     [model.get_sites_cart().as_numpy_array() for model
  #      in model_list2])
  #   assert list2_xyz.shape[0] == len(list2_index)
  #
  #   assignments = []
  #
  #   tree = KDTree(list2_xyz)
  #   for model in model_list1:
  #     dists, inds = tree.query(model.get_sites_cart().as_numpy_array(), k=1)
  #     dists, inds = dists[:, 0], inds[:, 0]
  #     nearest_in_list2 = list2_index[inds]
  #     uniq, counts = np.unique(nearest_in_list2, return_counts=True)
  #     assignment = uniq[np.argmax(counts)]
  #     assignments.append(assignment)
  #   print(assignments)
  #   assert (len(assignments) == len(set(
  #     assignments))), "Unable to uniqely pairs models from two lists"
  #
  #   ret = [(model1, model_list2[assignments[i]]) for i, model1 in
  #          enumerate(model_list1)]
  #   return ret

  def compose_models(self, models):
    # compose a list of models into a single model. Renumber atom serial numbers
    assert len(models) > 1
    composed_model = models[0].deep_copy()
    new_h = None
    for c in composed_model.get_hierarchy().models()[0].chains():
      if new_h is None:
        new_h = iotbx.pdb.hierarchy.new_hierarchy_from_chain(c)
      else:
        c_detach = c.detached_copy()
        new_h.models()[0].append_chain(c_detach)

    for copy_model in models[1:]:
      for m in copy_model.get_hierarchy().models():
        for c in m.chains():
          c_detach = c.detached_copy()
          new_h.models()[0].append_chain(c_detach)

    # renumber atom serial
    i = 1
    for atom in new_h.atoms():
      atom.serial = i
      i += 1

    composed_model = mmtbx.model.manager(
      model_input=None,
      pdb_hierarchy=new_h,
      crystal_symmetry=composed_model.crystal_symmetry(),
    )
    composed_model.get_hierarchy().atoms().reset_i_seq()
    return composed_model

  def take_ncs_from_map(self, mmm, ncs_group, eps_rot=1e-03, eps_tran=1):
    """
    Extract the transformations from a map and assign them to an ncs_group
    object. This useful to make sure the transformations are a precise point
    symmetry from the cryo-EM reconstruction.

    Parameters
    ----------
    mmm : map_model_manager
    ncs_group : mmtbx.ncs.ncs_restraints_group_list.NCS_restraint_group
    eps_rot : tolerance to consider rotation matrices equivalent between what was
              found by iotbx.ncs.input and mmm.get_ncs_from_map()
    eps_tran : tolerance for translations

    Returns
    -------
    ncs_group : mmtbx.ncs.ncs_restraints_group_list.NCS_restraint_group,
                a modified ncs_group containing the map transformations
    """
    ncs_map = mmm.get_ncs_from_map()
    assert ncs_map is not None, "Unable to retrieve NCS from map."
    assert (
      len(ncs_map.ncs_groups()) == 1
    ), "Detected multiple NCS groups from map."
    ncs_group_map = ncs_map.ncs_groups()[0]
    unit_rot = rec((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (3, 3))
    unit_tran = rec((0.0, 0.0, 0.0), (3, 1))

    rots = ncs_group_map.rota_matrices()
    trans = ncs_group_map.translations_orth()

    assert approx_equal(
      unit_rot, rots[0], eps=1e-03
    ), "First NCS rotation from map not a reference rotation"
    assert approx_equal(
      unit_tran, trans[0], eps=1e-03
    ), "First NCS translation from map not a reference translation"

    # establish the relationship between the map ncs and the model ncs objects
    pairing = [
      (copy, None) for copy in ncs_group.copies
    ]  # (copy object,index of ncs_map transformation)

    for i, (copy, match) in enumerate(pairing):
      for j, (rot, tran) in enumerate(zip(rots, trans)):
        matched_rot = approx_equal(copy.r, rot, eps=eps_rot, out=self.devnull)
        matched_tran = approx_equal(
          copy.t, tran, eps=eps_tran, out=self.devnull
        )
        if matched_rot and matched_tran:
          assert (
            match is None
          ), "Unable to uniquely pair ncs transforms from map and model"
          pairing[i] = (copy, j)

    for copy, match in pairing:
      assert (
        match is not None
      ), "Unable to uniquely pair ncs transforms from map and model"

    for copy, match in pairing:
      copy.r = rots[match]
      copy.t = trans[match]

    return ncs_group

  @staticmethod
  def chain_id_generator(blacklist=[], out=None):
    all_chain_ids_list = all_chain_ids()
    i = -1
    while True:
      i += 1
      if i >= len(all_chain_ids_list):
        print("Exceeded max number of chains", file=out)
        break
      else:
        chain_id = None
        while True:
          if i >= len(all_chain_ids_list):
            print("Exceeded max number of chains", file=out)
            break
          chain_id = all_chain_ids_list[i]
          if chain_id not in blacklist:
            break
          else:
            i += 1
        yield chain_id

  @staticmethod
  def next_free_chain(model, blacklist=[]):

    if isinstance(model, list):
      models = model
    else:
      models = [model]
    chain_ids_taken = []
    for model in models:
      for chain in model.get_hierarchy().chains():
        chain_ids_taken.append(chain.id)
    unique_taken = list(set(chain_ids_taken))
    chain_gen = SymmetrizeSolvent.chain_id_generator(blacklist=blacklist)
    for chain_id in chain_gen:
      if chain_id not in unique_taken:
        return chain_id

  @staticmethod
  def chain_ids(model):
    return list(set([c.id for c in model.get_hierarchy().chains()]))

  @staticmethod
  def n_chains(model):
    by_id = len(SymmetrizeSolvent.chain_ids(model))
    by_entity = 0
    for m in model.get_hierarchy().models():
      for c in m.chains():
        by_entity += 1
    return by_id, by_entity


# Function below selects solvent as single atom residue groups. I think it is
# better now to use explicit selection string, but need to ask
#
# def select_solvent(model,
#                    water_only=False,
#                    solvent_str_selection=None):
#   """
#   Select all solvent from model. Can choose water_only or manually
#   specify solvent string. Otherwise solvent is defined as all single atom
#   residue groups that are not nucleotide or protein.
#
#   Parameters
#   ----------
#   model : mmtbx.model.model.manager
#   return_str_selection : Whether to build a selection string for solvent
#
#   Returns
#   -------
#   solvent_selection, solvent_iselection : flex arrays selecting solvent
#   Optional: solvent_str_selection : a string selecting solvent in model
#   """
#   if solvent_str_selection is not None:
#     solvent_iselection = model.iselection(solvent_str_selection)
#     solvent_selection = flex.bool(model.get_number_of_atoms(), False)
#     solvent_selection = solvent_selection.set_selected(solvent_iselection, True)
#   elif water_only:
#     solvent_str_selection = "water"
#     solvent_iselection = model.iselection(solvent_str_selection)
#     solvent_selection = flex.bool(model.get_number_of_atoms(), False)
#     solvent_selection = solvent_selection.set_selected(solvent_iselection, True)
#   else:
#     sel_str_other = "not (nucleotide or protein)"
#     other_iselection = model.iselection(sel_str_other)
#     other_model = model.select(other_iselection)
#
#     solvent_selection = flex.bool(len(other_iselection), False)
#
#
#     for m in other_model.get_hierarchy().models():
#         for c in m.chains():
#             for rg in c.residue_groups():
#                 atoms = rg.atoms()
#                 if len(atoms) == 1:
#                     atom = atoms[0]
#                     solvent_selection[atom.i_seq] = True
#
#
#     solvent_iselection = other_iselection.select(solvent_selection)
#     solvent_selection = flex.bool(model.get_number_of_atoms(), False)
#     solvent_selection = solvent_selection.set_selected(solvent_iselection, True)
#
#     return solvent_selection, solvent_iselection
