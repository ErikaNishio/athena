//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../crdiffusion/crdiffusion.hpp"
#include "../crdiffusion/mg_crdiffusion.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"


#if !CRDIFFUSION_ENABLED
#error "The implicit CR diffusion solver must be enabled (-crdiff)."
#endif

namespace {
  int NECRbin;
  Real e0;
  Real E_k_min,E_k_max;

  // dimension-less constants
  constexpr Real four_pi_G = 1.0;
  constexpr Real rc = 6.45; // the BE radius in the normalized unit system
  constexpr Real rcsq = 26.0 / 3.0;      // the parameter of the BE profile
  constexpr Real bemass = 197.561;       // the total mass of the critical BE sphere

  // dimensional constants
  constexpr Real pi   = M_PI;
  constexpr Real cs10 = 1.9e4;        // sound speed at 10K, cm / s
  constexpr Real msun = 1.9891e33;    // solar mass, g
  constexpr Real pc   = 3.0857000e18; // parsec, cm
  constexpr Real au   = 1.4959787e13; // astronomical unit, cm
  constexpr Real yr   = 3.15569e7;    // year, s
  constexpr Real G    = 6.67259e-8;   // gravitational constant, dyn cm^2 g^-2
  constexpr Real c    = 2.99792458e10;// light speed
  constexpr Real mH2 = 1.67e-24*2.0;   // H_2 mass
  
  constexpr Real eV   = 1.6022e-12;   // from eV to erg

  // units in cgs
  Real m0, v0, t0, l0, rho0, gauss;

  // parameters and derivatives
  Real mass, temp, f, rhocrit, omega, bz, mu, amp,zeta;

  AthenaArray<Real> *pecr0,*pEk0,*pD_para0,*pLambda0,*pzeta_factor0;
}

Real CR_spectra_L(Real E_k){
  //Ivlev+2015 proton model L
  Real C = 2.4e15;
  Real E0 = 5.0e8; //500MeV = 5e8
  Real alpha = 0.1;
  Real beta = 2.8;
  return C*std::pow(E_k,alpha)/std::pow(E_k+E0,beta);
}

Real CR_spectra_H(Real E_k){
  //Ivlev+2015 proton model H
  Real C = 2.4e15;
  Real E0 = 5.0e8;
  Real alpha = -0.8;
  Real beta = 1.9;
  return C*std::pow(E_k,alpha)/std::pow(E_k+E0,beta);
}

Real pk(Real E_k){
  Real mp = 931.4941024e6; //proton mass energy
  Real pk = std::sqrt(E_k*(E_k+2.0*mp))*eV/c; //CR kinetic momentum

  return pk;
}

Real D_para(Real E_k){
  Real D;
  Real D_n = 1e28; //cm^2s^-1 diffusion coefficient of 10GeV Nava & Gabici 2013
  if (E_k > 1e9){
      D = D_n*std::pow(pk(E_k)/pk(1e10),1.0/3.0);
  }else{
      D = (D_n*std::pow(pk(1e9)/pk(1e10),1.0/3.0))*std::pow(pk(E_k)/pk(1e9),4.0/3.0);
  }
  
  return D;
}

Real zeta_factor(Real E_k,Real L_loss,Real Lambda){
  //cross section between H_2 and p_cr (Padovani 2009)
  Real a0 = 5.29177210544e-9;
  Real I = 13.598;
  Real x = E_k/1840.0/I;
  Real A = 0.71;
  Real B = 1.63;
  Real C = 0.51;
  Real D = 1.24;
  Real sigma1 = 4.0*pi*SQR(a0)*C*pow(x,D);
  Real sigma2 = 4*pi*SQR(a0)*(A*std::log(1.0+x)+B)/x;
  Real sigma_p = 1.0/(1.0/sigma1+1.0/sigma2);

  Real eps = L_loss/sigma_p; // loss energy of proton per ionization event

  Real zeta_f = Lambda/eps;

  return zeta_f;

}

void CR_value_calculation(int nvar){
  AthenaArray<Real> &ecr0 = *pecr0;
  AthenaArray<Real> &Ek0 = *pEk0;
  AthenaArray<Real> &D_para0 = *pD_para0;
  AthenaArray<Real> &Lambda0 = *pLambda0;
  AthenaArray<Real> &zeta_factor0 = *pzeta_factor0;

  //load attenuation function data
  FILE *fp;

    fp = fopen("/Users/erika/project/D_para_Lambda_function/p_lossfunction.dat","r");
    if (fp == NULL) {
        printf("Error: cannot open the loss function data\n");
    }
    Real Ek,Lk,Lambda_k;
    int imax = 1000;//number of array
    Real Ek_table[imax],Loss_function_table[imax],Lambda_table[imax];

    for(int k=0;k<imax;k++){
        Ek_table[k] = 0.0;
        Loss_function_table[k] = 0.0;
        Lambda_table[k] = 0.0;
    }

    int n_table_end = 0;

    //real file
    while(fscanf(fp, "%le %le %le",&Ek,&Lk, &Lambda_k) != EOF) {
        Ek_table[n_table_end] = Ek;
        Loss_function_table[n_table_end] = Lk;
        Lambda_table[n_table_end] = Lambda_k;

        n_table_end += 1;
    }

  fclose(fp);

  Real E_k_start = std::log10(E_k_min);
  Real E_k_end = std::log10(E_k_max);

  Real dE_k_bin = (E_k_end - E_k_start)/(2.0*nvar);

  int num = 1000;
  Real dE_k = (10.0 - E_k_start)/(2.0*num);//The maximum value of the integral is taken up to 10 GeV.

  Real e_cr,E_k,E_k_bin_sl,E_k_bin_sr,E_k_sl,E_k_sr;
  Real lambda,loss_fuction,zeta_f;
  Real zeta_all = 0.0;
  for (int n=0; n < nvar; n++){
    E_k_bin_sl = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n));
    E_k_bin_sr = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n+2.0));
    Ek = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n+1.0));
    Ek0(n) = Ek;
    D_para0(n) = D_para(Ek)/l0/l0*t0;

    //calculate Lambda_unction & zeta_factor
    for(int i=0; i < n_table_end-1; i++){
      if(Ek_table[i] < Ek && Ek <= Ek_table[i+1]){
        lambda = pow(10.0,(std::log10(Ek/Ek_table[i])*std::log10(Lambda_table[i])
                 +std::log10(Ek_table[i+1]/Ek)*std::log10(Lambda_table[i+1]))
                 /(std::log10(Ek_table[i+1]/Ek_table[i])));
        loss_fuction = pow(10.0,(std::log10(Ek/Ek_table[i])*std::log10(Loss_function_table[i])
                 +std::log10(Ek_table[i+1]/Ek)*std::log10(Loss_function_table[i+1]))
                 /(std::log10(Ek_table[i+1]/Ek_table[i])));
        zeta_f = zeta_factor(Ek,loss_fuction,lambda);

        Lambda0(n) = lambda/mH2*rho0*t0;//
        zeta_factor0(n) = zeta_f*e0;
        break;
      }
    }

    e_cr = 0.0;
      for(int i = 0; i < num; i++){
        E_k = E_k_min*std::pow(10.0,dE_k*(2.0*i+1.0));
        E_k_sl = E_k_min*std::pow(10.0,dE_k*(2.0*i));
        E_k_sr = E_k_min*std::pow(10.0,dE_k*(2.0*i+2.0));
        if(E_k_bin_sl < E_k_sl && E_k_sr <= E_k_bin_sr){
          e_cr += CR_spectra_L(E_k)*(E_k_sr-E_k_sl);
        }else if(E_k_sr > E_k_bin_sr){
          ecr0(n) = e_cr/e0;
          zeta_all += ecr0(n)*zeta_factor0(n);
          break;
        }
      }
    printf("%e,%e,%e,%e\n",E_k,D_para0(n),Lambda0(n),zeta_all);
  }
}



void CRFixedInnerX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++)
          dst(n,k,j,is-i-1) = 2.0*ecr0(n)- dst(n,k,j,is);
      }
    }
  }
  return;
}

void CRFixedOuterX1(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=0; i<ngh; i++)
          dst(n,k,j,ie+i+1) = 2.0*ecr0(n) - dst(n,k,j,ie);
      }
    }
  }
  return;
}

void CRFixedInnerX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++)
          dst(n,k,js-j-1,i) = 2.0*ecr0(n) - dst(n,k,js,i);
      }
    }
  }
  return;
}

void CRFixedOuterX2(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=ks; k<=ke; k++) {
      for (int j=0; j<ngh; j++) {
        for (int i=is; i<=ie; i++)
          dst(n,k,je+j+1,i) = 2.0*ecr0(n) - dst(n,k,je,i);
      }
    }
  }
  return;
}

void CRFixedInnerX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++)
          dst(n,ks-k-1,j,i) = 2.0*ecr0(n) - dst(n,ks,j,i);
      }
    }
  }
  return;
}

void CRFixedOuterX3(AthenaArray<Real> &dst, Real time, int nvar,
                    int is, int ie, int js, int je, int ks, int ke, int ngh,
                    const MGCoordinates &coord) {
  AthenaArray<Real> &ecr0 = *pecr0;
  for (int n=0; n<nvar; n++) {
    for (int k=0; k<ngh; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++)
          dst(n,ke+k+1,j,i) = 2.0*ecr0(n) - dst(n,ke,j,i);
      }
    }
  }
  return;
}


int AMRCondition(MeshBlock *pmb) {
  if (pmb->block_size.x1min >= 0.25 && pmb->block_size.x1min <=0.251
  &&  pmb->block_size.x2min >= 0.25 && pmb->block_size.x2min <=0.251
  &&  pmb->block_size.x3min >= 0.25 && pmb->block_size.x3min <=0.251) {
    if (pmb->pmy_mesh->ncycle >= pmb->loc.level - 1)
      return 1;
  }
  return 0;
}


//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  SetFourPiG(four_pi_G); // 4piG = 1.0
  mass = pin->GetReal("problem", "mass"); // solar mass
  temp = pin->GetReal("problem", "temperature");
  f = pin->GetReal("problem", "f"); // Density enhancement factor; f = 1 is critical
  amp = pin->GetOrAddReal("problem", "amp", 0.0); // perturbation amplitude
  mu = pin->GetOrAddReal("problem", "mu", 0.0); // micro gauss
  m0 = mass * msun / (bemass*f); // total mass = 1.0
  v0 = cs10 * std::sqrt(temp/10.0); // cs at 10K = 1.0
  rho0 = (v0*v0*v0*v0*v0*v0) / (m0*m0) /(64.0*pi*pi*pi*G*G*G);
  t0 = 1.0/std::sqrt(4.0*pi*G*rho0); // sqrt(1/4pi G rho0) = 1.0
  l0 = v0 * t0;
  gauss = std::sqrt(rho0*v0*v0*4.0*pi);
  Real mucrit1 = 0.53/(3.0*pi)*std::sqrt(5.0/G);
  bz = mass*msun/mucrit1/mu/pi/SQR(rc*l0)/gauss;
  e0 = eV/rho0/v0/v0;

  NECRbin = pin->GetReal("crdiffusion", "NECRbin");
  E_k_min = 1e7;
  E_k_max = 3e8;

  AllocateRealUserMeshDataField(5);
  ruser_mesh_data[0].NewAthenaArray(NECRbin);//ecr
  ruser_mesh_data[1].NewAthenaArray(NECRbin);//Ek_bin
  ruser_mesh_data[2].NewAthenaArray(NECRbin);//D_para
  ruser_mesh_data[3].NewAthenaArray(NECRbin);//Lambda
  ruser_mesh_data[4].NewAthenaArray(NECRbin);//zeta_factor

  pecr0 = &(ruser_mesh_data[0]);
  pEk0 = &(ruser_mesh_data[1]);
  pD_para0 = &(ruser_mesh_data[2]);
  pLambda0 = &(ruser_mesh_data[3]);
  pzeta_factor0 = &(ruser_mesh_data[4]);

  CR_value_calculation(NECRbin);

  EnrollUserRefinementCondition(AMRCondition);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x1, CRFixedInnerX1);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x1, CRFixedOuterX1);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x2, CRFixedInnerX2);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x2, CRFixedOuterX2);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x3, CRFixedInnerX3);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x3, CRFixedOuterX3);
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief cosmic ray diffusion test
//======================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real gamma = peos->GetGamma();
  Real r0 = 1e-3, rho = 1e-13/rho0;

  for(int k=ks; k<=ke; ++k) {
    Real x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      Real x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        Real r2 = SQR(x1)+SQR(x2)+SQR(x3);
        phydro->u(IDN,k,j,i) = rho/std::min(r2,r0*r0);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS)
          phydro->u(IEN,k,j,i) = phydro->u(IDN,k,j,i)/(gamma-1.0);
      }
    }
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je+1; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x2f(k,j,i) = 0.0;
          }
        }
      }
    }
    if (block_size.nx3 > 1) {
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            pfield->b.x3f(k,j,i) = bz;
          }
        }
      }
    }

    pfield->CalculateCellCenteredField(pfield->b,pfield->bcc,pcoord,is,ie,js,je,ks,ke);

    if (NON_BAROTROPIC_EOS) {
      for(int k=ks; k<=ke; ++k) {
        for(int j=js; j<=je; ++j) {
          for(int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) +=
              0.5*(SQR((pfield->bcc(IB1,k,j,i)))
                 + SQR((pfield->bcc(IB2,k,j,i)))
                 + SQR((pfield->bcc(IB3,k,j,i))));
          }
        }
      }
    }
  }

  //n個のec,D,Lを入れる
  AthenaArray<Real> &ecr0 = *pecr0;
  AthenaArray<Real> &D_para0 = *pD_para0;
  AthenaArray<Real> &Lambda0 = *pLambda0;
  AthenaArray<Real> &zeta_factor0 = *pzeta_factor0;

  for (int n=0; n < NECRbin; n++){
    pcrdiff->Lambda(n) = Lambda0(n);
    pcrdiff->zeta_factor(n) = zeta_factor0(n);

    for(int k=ks; k<=ke; ++k) {
      for(int j=js; j<=je; ++j) {
        for(int i=is; i<=ie; ++i){
          pcrdiff->Dpara(n,k,j,i) = D_para0(n);
          pcrdiff->Dperp(n,k,j,i) = D_para0(n)/100.0;
          pcrdiff->ecr(n,k,j,i) = ecr0(n);
        }
      }
    }
  }

  return;
}

