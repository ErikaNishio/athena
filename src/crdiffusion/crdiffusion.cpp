//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file crdiffusion.cpp
//! \brief implementation of functions in class CRDiffusion

// C headers

// C++ headers
#include <iostream>
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../bvals/bvals_interfaces.hpp"
#include "../bvals/cc/bvals_cc.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/buffer_utils.hpp"
#include "crdiffusion.hpp"
#include "mg_crdiffusion.hpp"


//----------------------------------------------------------------------------------------
//! \fn CRDiffusion::CRDiffusion(MeshBlock *pmb, ParameterInput *pin)
//! \brief CRDiffusion constructor
CRDiffusion::CRDiffusion(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb), 
    ecr1(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    ecr2(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    source1(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    source2(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff1(NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coeff2(NCOEFF, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    coarse_ecr(pmb->ncc3, pmb->ncc2, pmb->ncc1,
              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
               AthenaArray<Real>::DataStatus::empty)),
    empty_flux{AthenaArray<Real>(), AthenaArray<Real>(), AthenaArray<Real>()},
    output_defect(false), crbvar(pmb, &ecr, &coarse_ecr, empty_flux, false),
    refinement_idx_(), 
    Dpara1_(), Dperp1_(), Lambda1_(),
    Dpara2_(), Dperp2_(), Lambda2_() {
  Dpara1_ = pin->GetReal("crdiffusion", "Dpara1");
  Dperp1_ = pin->GetReal("crdiffusion", "Dperp1");
  Lambda1_ = pin->GetReal("crdiffusion", "Lambda1");
  Dpara2_ = pin->GetReal("crdiffusion", "Dpara2");
  Dperp2_ = pin->GetReal("crdiffusion", "Dperp2");
  Lambda2_ = pin->GetReal("crdiffusion", "Lambda2");

  output_defect = pin->GetOrAddBoolean("crdiffusion", "output_defect", false);
  if (output_defect)
    def.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);

  pmb->RegisterMeshBlockData(ecr);
  // "Enroll" in S/AMR by adding to vector of tuples of pointers in MeshRefinement class
  if (pmb->pmy_mesh->multilevel)
    refinement_idx_ = pmy_block->pmr->AddToRefinement(&ecr, &coarse_ecr);

  pmg = new MGCRDiffusion(pmb->pmy_mesh->pmcrd, pmb);

  // Enroll CellCenteredBoundaryVariable object
  crbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&crbvar);
  pmb->pbval->pcrbvar = &crbvar;
}


//----------------------------------------------------------------------------------------
//! \fn CRDiffusion::~CRDiffusion()
//! \brief CRDiffusion destructor
CRDiffusion::~CRDiffusion() {
  delete pmg;
}


//----------------------------------------------------------------------------------------
//! \fn void CRDiffusion::CalculateCoefficients()
//! \brief Calculate coefficients required for CR calculation
void CRDiffusion::CalculateCoefficients(const AthenaArray<Real> &w,
                                        const AthenaArray<Real> &bcc) {
  int il = pmy_block->is - NGHOST, iu = pmy_block->ie + NGHOST;
  int jl = pmy_block->js, ju = pmy_block->je;
  int kl = pmy_block->ks, ku = pmy_block->ke;
  if (pmy_block->pmy_mesh->f2)
    jl -= NGHOST, ju += NGHOST;
  if (pmy_block->pmy_mesh->f3)
    kl -= NGHOST, ku += NGHOST;
  Real Dpara1 = Dpara1_, Dperp1 = Dperp1_, Lambda1 = Lambda1_;
  Real Dpara2 = Dpara2_, Dperp2 = Dperp2_, Lambda2 = Lambda2_;

  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          const Real &bx = bcc(IB1,k,j,i);
          const Real &by = bcc(IB2,k,j,i);
          const Real &bz = bcc(IB3,k,j,i);
          Real ba = std::sqrt(SQR(bx) + SQR(by) + SQR(bz) + TINY_NUMBER);
          Real nx = bx / ba, ny = by / ba, nz = bz / ba;
          coeff1(DXX,k,j,i) = Dperp1 + (Dpara1 - Dperp1) * nx * nx;
          coeff1(DXY,k,j,i) =         (Dpara1 - Dperp1) * nx * ny;
          coeff1(DXZ,k,j,i) =         (Dpara1 - Dperp1) * nx * nz;
          coeff1(DYY,k,j,i) = Dperp1 + (Dpara1 - Dperp1) * ny * ny;
          coeff1(DYZ,k,j,i) =         (Dpara1 - Dperp1) * ny * nz;
          coeff1(DZZ,k,j,i) = Dperp1 + (Dpara1 - Dperp1) * nz * nz;
          coeff1(NLAMBDA, k,j,i) = Lambda1 * w(IDN,k,j,i);

          coeff2(DXX,k,j,i) = Dperp2 + (Dpara2 - Dperp2) * nx * nx;
          coeff2(DXY,k,j,i) =         (Dpara2 - Dperp2) * nx * ny;
          coeff2(DXZ,k,j,i) =         (Dpara2 - Dperp2) * nx * nz;
          coeff2(DYY,k,j,i) = Dperp2 + (Dpara2 - Dperp2) * ny * ny;
          coeff2(DYZ,k,j,i) =         (Dpara2 - Dperp2) * ny * nz;
          coeff2(DZZ,k,j,i) = Dperp2 + (Dpara2 - Dperp2) * nz * nz;
          coeff2(NLAMBDA, k,j,i) = Lambda2 * w(IDN,k,j,i);
        }
      }
    }
  } else {
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          coeff(DXX,k,j,i) = coeff(DXY,k,j,i) = coeff(DXZ,k,j,i) = coeff(DYY,k,j,i)
                           = coeff(DYZ,k,j,i) = coeff(DZZ,k,j,i) = Dpara;
          coeff(NLAMBDA, k,j,i) = Lambda * w(IDN,k,j,i);
        }
      }
    }
  }

  return;
}


