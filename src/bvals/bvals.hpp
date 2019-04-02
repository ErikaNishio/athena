#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.hpp
//  \brief defines BoundaryBase, BoundaryValues classes used for setting BCs on all data

// C headers

// C++ headers
#include <string>   // string
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "./bvals_interfaces.hpp"

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// forward declarations
// TODO(felker): how many of these foward declarations are actually needed now?
// Can #include "./bvals_interfaces.hpp" suffice?
class Mesh;
class MeshBlock;
class MeshBlockTree;
class ParameterInput;
class Coordinates;
struct RegionSize;

// free functions to return boundary flag given input string, and vice versa
BoundaryFlag GetBoundaryFlag(const std::string& input_string);
std::string GetBoundaryString(BoundaryFlag input_flag);

//----------------------------------------------------------------------------------------
//! \class BoundaryBase
//  \brief Base class for all BoundaryValues classes (BoundaryValues and MGBoundaryValues)

class BoundaryBase {
 public:
  BoundaryBase(Mesh *pm, LogicalLocation iloc, RegionSize isize,
               BoundaryFlag *input_bcs);
  virtual ~BoundaryBase();
  // 1x pair (neighbor index, buffer ID) per entire SET of separate variable buffers
  // (Hydro, Field, Passive Scalar, Gravity, etc.). Greedy allocation for worst-case
  // of refined 3D; only 26 entries needed/initialized if unrefined 3D, e.g.
  static NeighborIndexes ni[56];
  static int bufid[56];

  NeighborBlock neighbor[56];
  int nneighbor;
  int nblevel[3][3][3];
  LogicalLocation loc;
  BoundaryFlag block_bcs[6];
  PolarNeighborBlock *polar_neighbor_north, *polar_neighbor_south;

  static int CreateBvalsMPITag(int lid, int bufid, int phys);
  static int CreateBufferID(int ox1, int ox2, int ox3, int fi1, int fi2);
  static int BufferID(int dim, bool multilevel);
  static int FindBufferID(int ox1, int ox2, int ox3, int fi1, int fi2);

  void SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist, int *nslist);

 protected:
  // 1D refined or unrefined=2
  // 2D refined=12, unrefined=8
  // 3D refined=56, unrefined=26. Refinement adds: 3*6 faces + 1*12 edges = +30 neighbors
  static int maxneighbor_;

  Mesh *pmy_mesh_;
  RegionSize block_size_;
  AthenaArray<Real> sarea_[2];

 private:
  // calculate 3x shared static data members when constructing only the 1st class instance
  // int maxneighbor_=BufferID() computes ni[] and then calls bufid[]=CreateBufferID()
  static bool called_;
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValues
//  \brief centralized class for interacting with each individual variable boundary data

class BoundaryValues : public BoundaryBase, //public BoundaryPhysics,
                       public BoundaryCommunication {
 public:
  BoundaryValues(MeshBlock *pmb, BoundaryFlag *input_bcs, ParameterInput *pin);
  ~BoundaryValues();

  // inherited functions (interface shared with BoundaryVariable objects):
  // ------
  // called before time-stepper:
  void SetupPersistentMPI() final; // setup MPI requests

  // called during time-stepper:
  void StartReceiving(BoundaryCommSubset phase) final;
  void ClearBoundary(BoundaryCommSubset phase) final;

  // non-inhertied / unique functions (do not exist in BoundaryVariable objects):
  // (these typically involve a coupled interaction of boundary variable/quantities)
  // ------
  void ApplyPhysicalBoundaries(const Real time, const Real dt);
  void ProlongateBoundaries(const Real time, const Real dt);
  // above function wraps the following S/AMR-operations (within a loop over nneighbor):
  // (the next function is also called within 3x nested loops over nk,nj,ni)
  void RestrictGhostCellsOnSameLevel(const NeighborBlock& nb, int nk, int nj, int ni);
  void ApplyPhysicalBoundariesOnCoarseLevel(
      const NeighborBlock& nb, const Real time, const Real dt,
      int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateGhostCells(const NeighborBlock& nb,
                            int si, int ei, int sj, int ej, int sk, int ek);

  // check safety of user's configuration in Mesh::Initialize before SetupPersistentMPI()
  void CheckBoundary();
  void CheckPolarBoundaries();

  // variable-length arrays of references to BoundaryVariable instances
  // containing all BoundaryVariable instances:
  std::vector<BoundaryVariable *> bvars;
  // subset of bvars that are exchanged in the main TimeIntegratorTaskList
  std::vector<BoundaryVariable *> bvars_main_int;

  // function for distributing unique "phys" bitfield IDs to BoundaryVariable objects and
  // other categories of MPI communication for generating unique MPI_TAGs
  int ReserveTagVariableIDs(int num_phys);

 private:
  MeshBlock *pmy_block_;  // ptr to MeshBlock containing this BoundaryValues
  int num_north_polar_blocks_, num_south_polar_blocks_;
  int nface_, nedge_;

  // For spherical polar coordinates edge-case: if one MeshBlock wraps entirely around
  // the pole (azimuthally), shift the k-axis by nx3/2 for cell- and face-centered
  // variables, & emf, using this temporary 1D array.
  AthenaArray<Real> azimuthal_shift_;

  // Store signed, but positive, integer corresponding to the next unused value to be used
  // as unique ID for a BoundaryVariable object's single set of MPI calls (formerly "enum
  // AthenaTagMPI"). 5 bits of unsigned integer representation are currently reserved
  // for this "phys" part of the bitfield tag, making 0, ..., 31 legal values
  int bvars_next_phys_id_; // should be identical for all BVals, MeshBlocks, MPI ranks

  BValFunc BoundaryFunction_[6];

  // Shearingbox (shared with Field and Hydro)
  // ShearingBoundaryBlock shbb_;  // shearing block properties: lists etc.
  // Real x1size_,x2size_,x3size_; // mesh_size.x1max-mesh_size.x1min etc. [Lx,Ly,Lz]
  // Real Omega_0_, qshear_;       // orbital freq and shear rate
  // int ShBoxCoord_;              // shearcoordinate type: 1 = xy (default), 2 = xz
  // int joverlap_;                // # of cells the shear runs over one block
  // Real ssize_;                  // # of ghost cells in x-z plane
  // Real eps_;                    // fraction part of the shear

  // int  send_inner_gid_[4], recv_inner_gid_[4]; // gid of meshblocks for communication
  // int  send_inner_lid_[4], recv_inner_lid_[4]; // lid of meshblocks for communication
  // int send_inner_rank_[4],recv_inner_rank_[4]; // rank of meshblocks for communication
  // int  send_outer_gid_[4], recv_outer_gid_[4]; // gid of meshblocks for communication
  // int  send_outer_lid_[4], recv_outer_lid_[4]; // lid of meshblocks for communication
  // int send_outer_rank_[4],recv_outer_rank_[4]; // rank of meshblocks for communication

  // temporary--- Added by @tomidakn on 2015-11-27 in f0f989f85f
  // TODO(KGF): consider removing this friendship designation
  friend class Mesh;
  // currently, this class friendship are required for copying send/recv buffers between
  // BoundaryVariable objects within different MeshBlocks on the same MPI rank:
  friend class BoundaryVariable;
  // TODO(KGF): consider removing these friendship designations
  friend class CellCenteredBoundaryVariable;
  friend class FaceCenteredBoundaryVariable;
};
#endif // BVALS_BVALS_HPP_
