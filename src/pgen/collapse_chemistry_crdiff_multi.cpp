//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file collapse.cpp
//! \brief Problem generator for collapse of a Bonnor-Ebert like sphere with AMR or SMR

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mg_gravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../crdiffusion/crdiffusion.hpp"
#include "../crdiffusion/mg_crdiffusion.hpp"
#include "../bvals/bvals.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"

#if SELF_GRAVITY_ENABLED != 2
#error "This problem generator requires Multigrid gravity solver."
#endif

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#if !CRDIFFUSION_ENABLED
#error "The implicit CR diffusion solver must be enabled (-crdiff)."
#endif

namespace {


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
constexpr Real eV   = 1.6022e-12;   // erg , eV
constexpr Real c    = 2.99792458e10;// light speed
constexpr Real mH2 = 1.67e-24*2.0;   // H_2 mass

// units in cgs
Real m0, v0, t0, l0, rho0, gauss;

// parameters and derivatives
Real mass, temp, f, rhocrit, omega, bz, mu, amp,zeta;

// AMR parameter
Real njeans; // Real is used intentionally

Real totalm;

//chemistry file name
std::string tn_sigmao_t;
std::string tn_sigmah_t;
std::string tn_sigmap_t;
std::string tn_sigmao_nt;
std::string tn_sigmah_nt;
std::string tn_sigmap_nt;

const int nrhodef  = 171;
const int ntempdef = 51;
const int nrhobdef = 171;
const int nrhozdef = 171;
const Real rhomindef  = -22.0;
const Real drho       = 0.1;
const Real tempmindef = 1.0;
const Real dtemp      = 0.05;
const Real rhobmindef = -16.0;
const Real drhob      = 0.1;
const Real rhozmindef = -5.0;
const Real drhoz      = 0.1;
int nrho, ntemp, nrhob, nrhoz;
Real rhomin, rhomax, tempmin, tempmax, rhobmin, rhobmax, rhozmin, rhozmax;
AthenaArray<Real> sigmaOT, sigmaONT, sigmaHT, sigmaHNT, sigmaPT, sigmaPNT;

Real rhou,tempu,bu,resui;

Real d_out;

int NECRbin;
Real e0;
Real E_k_min,E_k_max;
Real zeta_normalize;
Real zeta_penetration;

AthenaArray<Real> *pecr0,*pEk0,*pD_para0,*pLambda0,*pzeta_factor0;
} // namespace

// Mask the density outside the initial sphere
void SourceMask(AthenaArray<Real> &src, int is, int ie, int js, int je,
                int ks, int ke, const MGCoordinates &coord) {
  const Real rc2 = rc*rc;
  for (int k=ks; k<=ke; ++k) {
    Real z = coord.x3v(k);
    for (int j=js; j<=je; ++j) {
      Real y = coord.x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real x = coord.x1v(i);
        Real r2 = x*x + y*y + z*z;
        if (r2 > rc2)
          src(k, j, i) = 0.0;
      }
    }
  }
  return;
}


void CRMask(AthenaArray<Real> &coeff, int is, int ie, int js, int je,
                int ks, int ke, const MGCoordinates &coord) {
  const Real rc2 = rc*rc;
  for (int n = 0; n < NECRbin; ++n){
    for (int k=ks; k<=ke; ++k) {
      Real z = coord.x3v(k);
      for (int j=js; j<=je; ++j) {
        Real y = coord.x2v(j);
        for (int i=is; i<=ie; ++i) {
          Real x = coord.x1v(i);
          Real r2 = x*x + y*y + z*z;
          if (r2 > rc2)
            coeff(n, NLAMBDA, k, j, i) = 0.0;
        }
      }
    }
  }
  return;
}


// AMR refinement condition
int JeansCondition(MeshBlock *pmb) {
  Real njmin = 1e300;

  const Real dx = pmb->pcoord->dx1f(0); // assuming uniform cubic cells
  if (MAGNETIC_FIELDS_ENABLED) {
    if (NON_BAROTROPIC_EOS) {
      const Real gamma = pmb->peos->GetGamma();
      const Real fac = 2.0 * pi / dx;
      for (int k = pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
        for (int j = pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
          for (int i = pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            Real v = std::sqrt(gamma * pmb->phydro->w(IPR,k,j,i)
                              / pmb->phydro->w(IDN,k,j,i))+
                   + std::sqrt((SQR(pmb->pfield->bcc(IB1,k,j,i))
                              + SQR(pmb->pfield->bcc(IB2,k,j,i))
                              + SQR(pmb->pfield->bcc(IB3,k,j,i)))
                              / pmb->phydro->w(IDN,k,j,i));
            Real nj = v / std::sqrt(pmb->phydro->w(IDN,k,j,i));
            njmin = std::min(njmin, nj);
          }
        }
      }
      njmin *= fac;
    } else {
      const Real cs = pmb->peos->GetIsoSoundSpeed();
      const Real fac = 2.0 * pi / dx;
      for (int k = pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
        for (int j = pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
          for (int i = pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            Real v = cs + std::sqrt((SQR(pmb->pfield->bcc(IB1,k,j,i))
                                   + SQR(pmb->pfield->bcc(IB2,k,j,i))
                                   + SQR(pmb->pfield->bcc(IB3,k,j,i)))
                                   / pmb->phydro->w(IDN,k,j,i));
            Real nj = v / std::sqrt(pmb->phydro->w(IDN,k,j,i));
            njmin = std::min(njmin, nj);
          }
        }
      }
      njmin *= fac;
    }
  } else {
    if (NON_BAROTROPIC_EOS) {
      const Real gamma = pmb->peos->GetGamma();
      const Real fac = 2.0 * pi * std::sqrt(gamma) / dx;
      for (int k = pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
        for (int j = pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
          for (int i = pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            Real nj = std::sqrt(pmb->phydro->w(IPR,k,j,i))
                              / pmb->phydro->w(IDN,k,j,i);
            njmin = std::min(njmin, nj);
          }
        }
      }
      njmin *= fac;
    } else {
      const Real cs = pmb->peos->GetIsoSoundSpeed();
      const Real fac = 2.0 * pi / dx;
      for (int k = pmb->ks-NGHOST; k<=pmb->ke+NGHOST; ++k) {
        for (int j = pmb->js-NGHOST; j<=pmb->je+NGHOST; ++j) {
          for (int i = pmb->is-NGHOST; i<=pmb->ie+NGHOST; ++i) {
            Real nj = cs / std::sqrt(pmb->phydro->w(IDN,k,j,i));
            njmin = std::min(njmin, nj);
          }
        }
      }
      njmin *= fac;
    }
  }
  if (njmin < njeans)
    return 1;
  if (njmin > njeans * 2.5)
    return -1;
  return 0;
}

// Approximated Bonnor-Ebert profile
// Tomida 2011, PhD Thesis
Real BEProfile(Real r) {
  return std::pow(1.0+r*r/rcsq, -1.5);
}

void fileremove(const char *filename){
  FILE *fp;
  fp = fopen(filename,"r");
  if (fp!=NULL){
      fclose(fp);
      remove(filename);
  } 
  return;
}

void Cooling(MeshBlock *pmb, const Real time, const Real dt,
             const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
             const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
             AthenaArray<Real> &cons_scalar) {
  const Real gamma = pmb->peos->GetGamma();
  const Real gm1 = pmb->peos->GetGamma() - 1.0;
  const Real igm1 = 1.0 / gm1;

  // Fixed boundary condition outside the initial core
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    Real z = pmb->pcoord->x3v(k);
    for (int j=pmb->js; j<=pmb->je; ++j) {
      Real y = pmb->pcoord->x2v(j);
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real x = pmb->pcoord->x1v(i);
        Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        if (r > rc) {
          cons(IM1,k,j,i) = 0.0;
          cons(IM2,k,j,i) = 0.0;
          cons(IM3,k,j,i) = 0.0;
        }
      }
    }
  }

  return;
}

void MyBnd_ix1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh);

void MyBnd_ox1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke,int ngh);

void MyBnd_ix2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh);
            
void MyBnd_ox2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh);

void MyBnd_ix3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh);
              
void MyBnd_ox3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh);


void LoadConductivityTables(Real irhomin, Real irhomax, Real itempmin, Real itempmax,
                     Real irhobmin, Real irhobmax, Real irhozmin, Real irhozmax);
void DeleteConductivityTables(void);
void CalcDiffusivity(FieldDiffusion *pfdif, 
                     MeshBlock *pmb, 
                     const AthenaArray<Real> &w, const AthenaArray<Real> &bmag, 
                     int is, int ie, int js, int je, int ks, int ke);

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

//CR multi energy 関連の関数
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

Real linear_completion(Real x,Real x0,Real x1,Real y0,Real y1){
  return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

void CR_value_calculation(int nvar){
  AthenaArray<Real> &ecr0 = *pecr0;
  AthenaArray<Real> &Ek0 = *pEk0;
  AthenaArray<Real> &D_para0 = *pD_para0;
  AthenaArray<Real> &Lambda0 = *pLambda0;
  AthenaArray<Real> &zeta_factor0 = *pzeta_factor0;

  //load attenuation function data
  FILE *fp;

    fp = fopen("/work/nishioer/p_lossfunction.dat","r");
    if (fp == NULL) {
        printf("Error: cannot open the loss function data\n");
    }
    Real Ek,Lk,Lambda_k;
    int imax = 200;//number of array
    std::vector<Real> Ek_table(imax,0.0),Loss_function_table(imax,0.0),Lambda_table(imax,0.0);

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

  Real e_cr,E_k_bin_c,E_k_bin_sl,E_k_bin_sr,E_k,E_k_sl,E_k_sr;
  Real lambda,loss_fuction,zeta_f;
  Real zeta_all = 0.0;
  zeta_penetration = 0.0;
  int p;
  for (int n=0; n < nvar; n++){
    E_k_bin_sl = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n));
    E_k_bin_sr = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n+2.0));
    E_k_bin_c = E_k_min*std::pow(10.0,dE_k_bin*(2.0*n+1.0));
    Ek0(n) = E_k_bin_c;
    D_para0(n) = D_para(E_k_bin_c)/l0/l0*t0;

    //calculate Lambda_unction & zeta_factor
    //代表値のE_kはセルの大きい方(E_k_bin_sr)で取る
    for(int i=0; i < n_table_end-1; i++){
      if(Ek_table[i] < E_k_bin_sr && E_k_bin_sr <= Ek_table[i+1]){
        lambda = pow(10.0,linear_completion(std::log10(E_k_bin_sr),std::log10(Ek_table[i]),std::log10(Ek_table[i+1]),
        std::log10(Lambda_table[i]),std::log10(Lambda_table[i+1])));
        loss_fuction = pow(10.0,linear_completion(std::log10(E_k_bin_sr),std::log10(Ek_table[i]),std::log10(Ek_table[i+1]),
        std::log10(Loss_function_table[i]),std::log10(Loss_function_table[i+1])));
        zeta_f = zeta_factor(E_k_bin_sr,loss_fuction,lambda);

        Lambda0(n) = lambda/mH2*rho0*t0;// Lambda*rho = s^{-1}
        zeta_factor0(n) = zeta_f*e0;
        if(n == nvar-1){
          p = i;//tableの最後の番号を残す
        }
        break;
      }
    }

    e_cr = 0.0;
    Real vk;  
    int a = 0;
    for(int i = 0; i < num; i++){
      E_k = E_k_min*std::pow(10.0,dE_k*(2.0*i+1.0));
      E_k_sl = E_k_min*std::pow(10.0,dE_k*(2.0*i));
      E_k_sr = E_k_min*std::pow(10.0,dE_k*(2.0*i+2.0));

      if(E_k_bin_sl < E_k_sl && E_k_sr <= E_k_bin_sr){
          vk = c*sqrt(1.0 - SQR(931.4941024e6/(931.4941024e6+E_k)));
          e_cr += (4.0*pi*E_k_sr*CR_spectra_L(E_k_sr)/vk)*(E_k_sr-E_k_sl);
        }else if(E_k_sr > E_k_bin_sr){
          if(n < nvar-1){
              ecr0(n) = e_cr/e0;
              zeta_all += ecr0(n)*zeta_factor0(n);
              break;
          }else{
            if(a == 0){
              ecr0(n) = e_cr/e0;
              zeta_all += e_cr/e0*zeta_factor0(n);
              a += 1;
            }
            //透過する電離率を計算する
            if(Ek_table[p] < E_k_sr && E_k_sr <= Ek_table[p+1]){
                lambda = pow(10.0,linear_completion(std::log10(E_k_sr),std::log10(Ek_table[p]),std::log10(Ek_table[p+1]),
                std::log10(Lambda_table[p]),std::log10(Lambda_table[p+1])));
                loss_fuction = pow(10.0,linear_completion(std::log10(E_k_sr),std::log10(Ek_table[p]),std::log10(Ek_table[p+1]),
                std::log10(Loss_function_table[p]),std::log10(Loss_function_table[p+1])));
                zeta_f = zeta_factor(E_k_sr,loss_fuction,lambda);
                zeta_penetration += (4.0*pi*E_k_sr*CR_spectra_L(E_k_sr)/vk)*(E_k_sr-E_k_sl)
                *zeta_factor(E_k_bin_sr,loss_fuction,lambda)*e0;
            }else if(Ek_table[p+1] < E_k_sr && E_k_sr <= Ek_table[p+2]){
              if(p+1 >= n_table_end){
                printf("Completion is not possible.");
              }
              p += 1;
              lambda = pow(10.0,linear_completion(std::log10(E_k_sr),std::log10(Ek_table[p]),std::log10(Ek_table[p+1]),
              std::log10(Lambda_table[p]),std::log10(Lambda_table[p+1])));
              loss_fuction = pow(10.0,linear_completion(std::log10(E_k_sr),std::log10(Ek_table[p]),std::log10(Ek_table[p+1]),
              std::log10(Loss_function_table[p]),std::log10(Loss_function_table[p+1])));
              zeta_f = zeta_factor(E_k_sr,loss_fuction,lambda);
              zeta_penetration += (4.0*pi*E_k_sr*CR_spectra_L(E_k_sr)/vk)*(E_k_sr-E_k_sl)
              *zeta_factor(E_k_bin_sr,loss_fuction,lambda)*e0;
            }
          }
        }
    }
  }
    zeta_all += zeta_penetration;

  //規格化
  zeta_penetration = zeta_penetration/zeta_all*zeta_normalize;
  for (int n=0; n < nvar; n++){
    ecr0(n) = ecr0(n)/zeta_all*zeta_normalize;
  }
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
  rhocrit = pin->GetReal("problem", "rhocrit") / rho0;
  totalm = 0.0;
  Real tff = std::sqrt(3.0/8.0/f)*pi;
  Real omegatff = pin->GetOrAddReal("problem", "omegatff", 0.0);
  omega = omegatff/tff;
  Real mucrit1 = 0.53/(3.0*pi)*std::sqrt(5.0/G);
  Real mucrit2 = 1.0/(2.0*pi*std::sqrt(G));
  bz = mass*msun/mucrit1/mu/pi/SQR(rc*l0)/gauss;
  Real bzug = bz*gauss*1e6;
  zeta = pin->GetReal("problem", "zeta");
  e0 = eV/rho0/v0/v0;

  NECRbin = pin->GetReal("crdiffusion", "NECRbin");
  E_k_min = pin->GetReal("crdiffusion", "Ek_min");
  E_k_max = pin->GetReal("crdiffusion", "Ek_max");

  zeta_normalize = pin->GetReal("crdiffusion", "zeta_normalize");

  d_out = 6.980244e-02;

  //CR multi energyの設定
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
  
  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
      << "---  Dimensional parameters of the simulation  ---" << std::endl
      << "Total mass          : " << mass      << " \t\t[Msun]" << std::endl
      << "Initial temperature : " << temp      << " \t\t[K]" << std::endl
      << "Sound speed         : " << v0        << " \t\t[cm s^-1]" << std::endl
      << "Central density     : " << rho0*f    << " \t[g cm^-3]" << std::endl
      << "Cloud radius        : " << rc*l0/au  << " \t\t[au]" << std::endl
      << "Free fall time      : " << tff*t0/yr << " \t\t[yr]" << std::endl
      << "Angular velocity    : " << omega/t0  << " \t[s^-1]" << std::endl
      << "Angular velocity    : " << omega/t0*yr << " \t[yr^-1]" << std::endl
      << "Magnetic field      : " << bzug      << " \t\t[uGauss]" << std::endl
      << "Density Enhancement : " << f         << std::endl 
      << "---   Normalization Units of the simulation    ---" << std::endl
      << "Mass                : " << m0        << " \t[g]" << std::endl
      << "Mass                : " << m0/msun   << " \t[Msun]" << std::endl
      << "Length              : " << l0        << " \t[cm]" << std::endl
      << "Length              : " << l0/au     << " \t\t[au]" << std::endl
      << "Length              : " << l0/pc     << " \t[pc]" << std::endl
      << "Time                : " << t0        << " \t[s]" << std::endl
      << "Time                : " << t0/yr     << " \t\t[yr]" << std::endl
      << "Velocity            : " << v0        << " \t\t[cm s^-1]" << std::endl
      << "Density             : " << rho0      << " \t[g cm^-3]" << std::endl
      << "Magnetic field      : " << gauss     << " \t[Gauss]" << std::endl << std::endl
      << "--- Dimensionless parameters of the simulation ---" << std::endl
      << "Total mass          : " << bemass*f  << std::endl
      << "Sound speed at " << temp << " K : "  << 1.0 << std::endl
      << "Central density     : " << 1.0       << std::endl
      << "Cloud radius        : " << rc        << std::endl
      << "Free fall time      : " << tff       << std::endl
      << "m=2 perturbation    : " << amp       << std::endl
      << "Omega * tff         : " << omegatff << std::endl
      << "Mass-to-flux ratio  : " << mu        << std::endl << std::endl;
  }

  EnrollUserMGGravitySourceMaskFunction(SourceMask);
  EnrollUserMGCRDiffusionCoefficientMaskFunction(CRMask);
  
  //cooling
  if (NON_BAROTROPIC_EOS)
    EnrollUserExplicitSourceFunction(Cooling);
  
  //AMR
  if (adaptive) {
    njeans = pin->GetReal("problem","njeans");
    EnrollUserRefinementCondition(JeansCondition);
  }
  
  //resistivity

  rhou=rho0;
  tempu=temp;
  bu=gauss;
  resui=t0/l0/l0;

  /*std::string tabdir_t = pin->GetString("problem","tabdir_t");//dat_0.1um_fdg1e-2を指定
  std::string tabdir_nt = pin->GetString("problem","tabdir_nt");//dat_0.1um_fdg1e-2*/


  std::string tabdir_t = "/work/nishioer/chemistry/datT_fdg1e-2";
  std::string tabdir_nt = "/work/nishioer/chemistry/dat_0.1um_fdg1e-2";

  tn_sigmao_t = tabdir_t + "/sigmaOc2_T.dat";
  tn_sigmah_t = tabdir_t + "/sigmaHc2_T.dat";
  tn_sigmap_t = tabdir_t + "/sigmaPc2_T.dat";
  tn_sigmao_nt = tabdir_nt + "/sigmaOc2_NT.dat";
  tn_sigmah_nt = tabdir_nt + "/sigmaHc2_NT.dat";
  tn_sigmap_nt = tabdir_nt + "/sigmaPc2_NT.dat";

  if(Globals::my_rank == 0 ) {
      printf("sigO taken from %s for thermal eq. and %s for nonthermal eq.\n",tn_sigmao_t.c_str(), tn_sigmao_nt.c_str());
      printf("sigH taken from %s for thermal eq. and %s for nonthermal eq.\n",tn_sigmah_t.c_str(), tn_sigmah_nt.c_str());
      printf("sigP taken from %s for thermal eq. and %s for nonthermal eq.\n",tn_sigmap_t.c_str(), tn_sigmap_nt.c_str());
  }

  LoadConductivityTables(-22.0, -5.0,  // log10(rho)
                          1.0, 3.5,    // log10(T)
                         -16.0, 1.0, // log10(rho/B)
                         -5, 12.0);   // log10(rho/zeta)

  EnrollFieldDiffusivity(CalcDiffusivity);

  //User定義の境界条件
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, MyBnd_ix1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, MyBnd_ox1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, MyBnd_ix2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, MyBnd_ox2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, MyBnd_ix3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, MyBnd_ox3);

  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x1, CRFixedInnerX1);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x1, CRFixedOuterX1);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x2, CRFixedInnerX2);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x2, CRFixedOuterX2);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::inner_x3, CRFixedInnerX3);
  EnrollUserMGCRDiffusionBoundaryFunction(BoundaryFace::outer_x3, CRFixedOuterX3);

  return;
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  //領域内のある値を抽出してoutput
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(6);//2つにわける。max
  ruser_meshblock_data[1].NewAthenaArray(13);//sum

  //磁気抵抗率(eta_ohm,eta_ad)を出す
  AllocateUserOutputVariables(2);

  return ;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real igm1 = 1.0 / (peos->GetGamma() - 1.0);
  Real dx = pcoord->dx1f(is);

  FILE *fp;

    fp = fopen("/work/nishioer/sph.dat","r");
    if (fp == NULL) {
        printf("Error: cannot open the file\n");
    }

    double r,d,fd,M,aved;
    int imax = 10000;//構造体の数
    std::vector<Real> ri(imax,0.0),di(imax,0.0);

    int ik=0;

    //ファイルから読み込む

    while(fscanf(fp, "%le %le %le %le %le",&r,&d, &fd, &M, &aved) != EOF) {
        ri[ik] = r;
        di[ik] = d;

        ik += 1;
    }

    fclose(fp);
    
    Real d0 = di[ik-1];//外側の密度
    Real radout = ri[ik-1];

    //外部密度と外部の速度を入れる
    for (int k=ks; k<=ke+1; k++) {
        for (int j=js; j<=je+1; j++) {
            for (int i=is; i<=ie+1; i++) {
                phydro->u(IDN,k,j,i) = d0;
                phydro->u(IM1,k,j,i) = 0.0;
                phydro->u(IM2,k,j,i) = 0.0;
                phydro->u(IM3,k,j,i) = 0.0;
            }
        }
    }

    //初期条件として設定していく
    for (int k=ks; k<=ke+1; k++) {
      Real z = pcoord->x3v(k);
        for (int j=js; j<=je+1; j++) {
          Real y = pcoord->x2v(j);
            for (int i=is; i<=ie+1; i++) {
                //密度を入れていく
                Real x = pcoord->x1v(i);//セルの中心
                Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));

                //radc-ri[l]が今までの最低値よりも小さければraを置き換えて一番小さいriとdiを出す。中央の値と最も近いri[l]を出す
                for (int l = 1; l<imax; l++){
                    if(r < radout){
                        if (ri[l] > r) {
                            Real rp = ri[l];
                            Real rm = ri[l-1];
                            Real dp = di[l];
                            Real dm = di[l-1];
                            Real drp = rp-r;
                            Real drm = r-rm;
                            Real dc = drm/(drp+drm)*dp+drp/(drp+drm)*dm;
                            phydro->u(IDN,k,j,i) = f*dc;
                            break;
                        }
                    }
                }
                
                Real phi = std::atan2(y,x);

                if (r < rc) {
                  //剛体回転
                  phydro->u(IM1,k,j,i) = -phydro->u(IDN,k,j,i)*omega*y*(1.0+0.01*std::cos(2.0*phi));
                  phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*omega*x*(1.0+0.01*std::cos(2.0*phi));
                  phydro->u(IM3,k,j,i) = 0.0;
                } else {
                  phydro->u(IM1,k,j,i) = 0.0;
                  phydro->u(IM2,k,j,i) = 0.0;
                  phydro->u(IM3,k,j,i) = 0.0;
                }
                if (NON_BAROTROPIC_EOS)
                  phydro->u(IEN,k,j,i) = igm1 * phydro->u(IDN,k,j,i) // c_s = 1
                                      + 0.5*(SQR(phydro->u(IM1,k,j,i))+SQR(phydro->u(IM2,k,j,i))
                                      + SQR(phydro->u(IM3,k,j,i)))/phydro->u(IDN,k,j,i);
            }
        }
    }

  // initialize interface B, uniform
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x2f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          pfield->b.x3f(k,j,i) = bz;
        }
      }
    }
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=is; i<=ie; ++i) {
            phydro->u(IEN,k,j,i) += 0.5*SQR(bz);
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

  //既にファイルがあるなら消去
  fileremove("data.dat");
  fileremove("data_bcc.dat");

}

void LoadConductivityTables(Real irhomin, Real irhomax, Real itempmin, Real itempmax,
                     Real irhobmin, Real irhobmax, Real irhozmin, Real irhozmax)
{
  Real smlf=0.00001;
  // set the table range, assuming exact values are specified.
  rhomin=irhomin, tempmin=itempmin, rhobmin=irhobmin, rhozmin=irhozmin;
  rhomax=irhomax, tempmax=itempmax, rhobmax=irhobmax, rhozmax=irhozmax;
  nrho=(int)((rhomax-rhomin+drho*smlf)/drho)+1;
  ntemp=(int)((tempmax-tempmin+dtemp*smlf)/dtemp)+1;
  nrhob=(int)((rhobmax-rhobmin+drhob*smlf)/drhob)+1;
  nrhoz=(int)((rhozmax-rhozmin+drhoz*smlf)/drhoz)+1;

  int osrho=(int)((rhomin-rhomindef+drho*smlf)/drho);
  int ostemp=(int)((tempmin-tempmindef+dtemp*smlf)/dtemp);
  int osrhob=(int)((rhobmin-rhobmindef+drhob*smlf)/drhob);
  int osrhoz=(int)((rhozmin-rhozmindef+drhoz*smlf)/drhoz);
  // slightly shift the maximum ranges
  rhomax-=drho*smlf, tempmax-=dtemp*smlf, rhobmax-=drhob*smlf, rhozmax-=drhoz*smlf;

  // allocate the table arrays
  sigmaOT.NewAthenaArray(ntemp, nrho);
  sigmaHT.NewAthenaArray(nrhob, ntemp, nrho);
  sigmaPT.NewAthenaArray(nrhob, ntemp, nrho);
  sigmaONT.NewAthenaArray(ntemp, nrhoz);
  sigmaHNT.NewAthenaArray(nrhob, ntemp, nrhoz);
  sigmaPNT.NewAthenaArray(nrhob, ntemp, nrhoz);

  AthenaArray<float> dbuf;
  dbuf.NewAthenaArray(nrhobdef, ntempdef, std::max(nrhodef,nrhozdef));
  IOWrapper fot, fht, fpt, font, fhnt, fpnt;

  // load the table in parallel
  fot.Open(tn_sigmao_t.c_str(), IOWrapper::FileMode::read);
  fot.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhodef, sizeof(float));
  fot.Close();
  for(int j=0; j<ntemp; j++) {
    for(int i=0; i<nrho; i++){
      sigmaOT(j, i)=dbuf(0, ostemp+j, osrho+i);
    }
  }
  fht.Open(tn_sigmah_t.c_str(), IOWrapper::FileMode::read);
  fht.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhodef, sizeof(float));
  fht.Close();
  for(int k=0; k<nrhob; k++) {
    for(int j=0; j<ntemp; j++) {
      for(int i=0; i<nrho; i++)
        sigmaHT(k, j, i)=dbuf(osrhob+k, ostemp+j, osrho+i);
    }
  }
  fpt.Open(tn_sigmap_t.c_str(), IOWrapper::FileMode::read);
  fpt.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhodef, sizeof(float));
  fpt.Close();
  for(int k=0; k<nrhob; k++) {
    for(int j=0; j<ntemp; j++) {
      for(int i=0; i<nrho; i++)
        sigmaPT(k, j, i)=dbuf(osrhob+k, ostemp+j, osrho+i);
    }
  }
  font.Open(tn_sigmao_nt.c_str(), IOWrapper::FileMode::read);
  font.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhozdef, sizeof(float));
  font.Close();
  for(int j=0; j<ntemp; j++) {
    for(int i=0; i<nrhoz; i++)
      sigmaONT(j, i)=dbuf(0, ostemp+j, osrhoz+i);
  }
  fhnt.Open(tn_sigmah_nt.c_str(), IOWrapper::FileMode::read);
  fhnt.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhozdef, sizeof(float));
  fhnt.Close();
  for(int k=0; k<nrhob; k++) {
    for(int j=0; j<ntemp; j++) {
      for(int i=0; i<nrhoz; i++)
        sigmaHNT(k, j, i)=dbuf(osrhob+k, ostemp+j, osrhoz+i);
    }
  }
  fpnt.Open(tn_sigmap_nt.c_str(), IOWrapper::FileMode::read);
  fpnt.Read_all(dbuf.data(), nrhobdef*ntempdef*nrhozdef, sizeof(float));
  fpnt.Close();
  for(int k=0; k<nrhob; k++) {
    for(int j=0; j<ntemp; j++) {
      for(int i=0; i<nrhoz; i++)
        sigmaPNT(k, j, i)=dbuf(osrhob+k, ostemp+j, osrhoz+i);
    }
  }

  dbuf.DeleteAthenaArray();
}

void DeleteConductivityTables(void)
{
  sigmaOT.DeleteAthenaArray();
  sigmaONT.DeleteAthenaArray();
  sigmaHT.DeleteAthenaArray();
  sigmaHNT.DeleteAthenaArray();
  sigmaPT.DeleteAthenaArray();
  sigmaPNT.DeleteAthenaArray();
}

void CalcDiffusivity(FieldDiffusion *pfdif, MeshBlock* pmb, const AthenaArray<Real> &w,
     const AthenaArray<Real> &bmag, int is, int ie, int js, int je, int ks, int ke)
{
  Real cfl_number = pmb->pmy_mesh->cfl_number;
  Real dt = pmb->pmy_mesh->dt;
  Real zeta_radio = 1e-18+zeta_penetration;

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
#pragma omp simd
      for(int i=is; i<=ie; i++) {
         // convert to cgs, calculate the parameters
         Real r=w(IDN,k,j,i)*rhou;
//         Real t=w(IPR,k,j,i)/w(IDN,k,j,i)*tempu;
         Real t=tempu*w(IPR,k,j,i)/w(IDN,k,j,i);
//         Real t=eps0*eps0/sqrt(pmb->pcoord->x1v(i))*tempu;
         Real b=std::max( bmag(k,j,i)*bu, 1e-10);
         Real rb=r/b;
         Real rz=r/(pmb->pcrdiff->zeta(k,j,i) + zeta_radio);
         // calculate the array indexes and weighting factors
         Real lr=std::max((std::min(log10(r),rhomax)-rhomin)/drho,0.0);
         int ir=(int)lr;
         Real wr=lr-ir;
         Real lt=std::max((std::min(log10(t),tempmax)-tempmin)/dtemp,0.0);
         int it=(int)lt;
         Real wt=lt-it;
         Real lrb=std::max((std::min(log10(rb),rhobmax)-rhobmin)/drhob,0.0);
         int irb=(int)lrb;
         Real wrb=lrb-irb;
         Real lrz=std::max((std::min(log10(rz),rhozmax)-rhozmin)/drhoz,0.0);
         int irz=(int)lrz;
         Real wrz=lrz-irz;
         Real wt00=(1.0-wr)*(1.0-wt),  wt01=wr*(1.0-wt),  wt10=(1.0-wr)*wt,  wt11=wr*wt;
         Real wn00=(1.0-wrz)*(1.0-wt), wn01=wrz*(1.0-wt), wn10=(1.0-wrz)*wt, wn11=wrz*wt;
         Real wt000=(1.0-wrb)*wt00, wt001=(1.0-wrb)*wt01;
         Real wt010=(1.0-wrb)*wt10, wt011=(1.0-wrb)*wt11;
         Real wt100=wrb*wt00, wt101=wrb*wt01, wt110=wrb*wt10, wt111=wrb*wt11;
         Real wn000=(1.0-wrb)*wn00, wn001=(1.0-wrb)*wn01;
         Real wn010=(1.0-wrb)*wn10, wn011=(1.0-wrb)*wn11;
         Real wn100=wrb*wn00, wn101=wrb*wn01, wn110=wrb*wn10, wn111=wrb*wn11;
         // calculate the conductivities interpolating the tables
         Real sgOT=sigmaOT(it,  ir)*wt00+sigmaOT(it,  ir+1)*wt01
                  +sigmaOT(it+1,ir)*wt10+sigmaOT(it+1,ir+1)*wt11;
         Real sgONT=sigmaONT(it,  irz)*wn00+sigmaONT(it,  irz+1)*wn01
                   +sigmaONT(it+1,irz)*wn10+sigmaONT(it+1,irz+1)*wn11;
         Real sgO=sgOT+sgONT;
         Real sgHT=sigmaHT(irb,  it,  ir)*wt000+sigmaHT(irb,  it,  ir+1)*wt001
                  +sigmaHT(irb,  it+1,ir)*wt010+sigmaHT(irb,  it+1,ir+1)*wt011
                  +sigmaHT(irb+1,it,  ir)*wt100+sigmaHT(irb+1,it,  ir+1)*wt101
                  +sigmaHT(irb+1,it+1,ir)*wt110+sigmaHT(irb+1,it+1,ir+1)*wt111;
         Real sgHNT=sigmaHNT(irb,  it,  irz)*wn000+sigmaHNT(irb,  it,  irz+1)*wn001
                   +sigmaHNT(irb,  it+1,irz)*wn010+sigmaHNT(irb,  it+1,irz+1)*wn011
                   +sigmaHNT(irb+1,it,  irz)*wn100+sigmaHNT(irb+1,it,  irz+1)*wn101
                   +sigmaHNT(irb+1,it+1,irz)*wn110+sigmaHNT(irb+1,it+1,irz+1)*wn111;
         Real sgH=sgHT+sgHNT;
         Real sgPT=sigmaPT(irb,  it,  ir)*wt000+sigmaPT(irb,  it,  ir+1)*wt001
                  +sigmaPT(irb,  it+1,ir)*wt010+sigmaPT(irb,  it+1,ir+1)*wt011
                  +sigmaPT(irb+1,it,  ir)*wt100+sigmaPT(irb+1,it,  ir+1)*wt101
                  +sigmaPT(irb+1,it+1,ir)*wt110+sigmaPT(irb+1,it+1,ir+1)*wt111;
         Real sgPNT=sigmaPNT(irb,  it,  irz)*wn000+sigmaPNT(irb,  it,  irz+1)*wn001
                   +sigmaPNT(irb,  it+1,irz)*wn010+sigmaPNT(irb,  it+1,irz+1)*wn011
                   +sigmaPNT(irb+1,it,  irz)*wn100+sigmaPNT(irb+1,it,  irz+1)*wn101
                   +sigmaPNT(irb+1,it+1,irz)*wn110+sigmaPNT(irb+1,it+1,irz+1)*wn111;
         Real sgP=sgPT+sgPNT;

         Real fac = 1.0;

         // set the resistivities
         pfdif->etaB(FieldDiffusion::DiffProcess::ohmic,k,j,i)=fac*std::max(resui/sgO, 0.0);
         pfdif->etaB(FieldDiffusion::DiffProcess::hall,k,j,i)=0.0;
         pfdif->etaB(FieldDiffusion::DiffProcess::ambipolar,k,j,i)=fac*std::max(resui*(sgP/(sgH*sgH+sgP*sgP)-1.0/sgO), 0.0);
      }
    }
  }
}

void MyBnd_ix1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

        for(int k=ks; k<=ke; ++k){
          for(int j=js; j<=je; ++j){
            for(int i=1; i<=(NGHOST); ++i){
              prim(IDN,k,j,is-i) = prim(IDN,k,j,is);
              prim(IVX,k,j,is-i) = prim(IVX,k,j,is) ;
              prim(IVY,k,j,is-i) =  prim(IVY,k,j,is);
              prim(IVZ,k,j,is-i) = prim(IVZ,k,j,is);
              prim(IPR,k,j,is-i) = prim(IPR,k,j,is);

            }
          }
        } 

        if (MAGNETIC_FIELDS_ENABLED) {
          for(int k=ks; k<=ke; ++k){
            for(int j=js; j<=je; ++j){
              for(int i=1; i<=(NGHOST); ++i){
                  b.x1f(k,j,is-i) = b.x1f(k,j,is);
                }
              }
            }

            for (int k=ks; k<=ke; ++k) {
              for (int j=js; j<=je+1; ++j) {
                for (int i=1; i<=(NGHOST); ++i) {
                    b.x2f(k,j,is-i) = b.x2f(k,j,is);
                  }
                }
              }
            

              for (int k=ks; k<=ke+1; ++k) {
                for (int j=js; j<=je; ++j) {
                  for (int i=1; i<=(NGHOST); ++i) {
                      b.x3f(k,j,is-i) = b.x3f(k,j,is);
                    }
                  }
                }
        }

      return;
    }

void MyBnd_ox1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

                for(int k=ks; k<=ke; ++k){
                  for(int j=js; j<=je; ++j){
                    for(int i=1; i<=(NGHOST); ++i){
                      prim(IDN,k,j,ie+i) = prim(IDN,k,j,ie) ;
                      prim(IVX,k,j,ie+i) = prim(IVX,k,j,ie);
                      prim(IVY,k,j,ie+i) = prim(IVY,k,j,ie);
                      prim(IVZ,k,j,ie+i) = prim(IVZ,k,j,ie);
                      prim(IPR,k,j,ie+i) = prim(IPR,k,j,ie);
                    }
                  }
                }  

                if (MAGNETIC_FIELDS_ENABLED) {
                  for(int k=ks; k<=ke; ++k){
                    for(int j=js; j<=je; ++j){
                      for(int i=1; i<=(NGHOST); ++i){
                          b.x1f(k,j,ie+1+i) = b.x1f(k,j,ie+1);
                        }
                      }
                    }

                    for (int k=ks; k<=ke; ++k) {
                      for (int j=js; j<=je+1; ++j) {
                        for (int i=1; i<=(NGHOST); ++i) {
                            b.x2f(k,j,ie+i) = b.x2f(k,j,ie);
                          }
                        }
                      }
                    

                      for (int k=ks; k<=ke+1; ++k) {
                        for (int j=js; j<=je; ++j) {
                          for (int i=1; i<=(NGHOST); ++i) {
                              b.x3f(k,j,ie+i) = b.x3f(k,j,ie);
                            }
                          }
                        }
                  }

      return;
      }

void MyBnd_ix2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

        for(int k=ks; k<=ke; ++k){
          for(int j=1; j<=(NGHOST); ++j){
            for(int i=is; i<=ie; ++i){
              prim(IDN,k,js-j,i) = prim(IDN,k,js,i);
              prim(IVX,k,js-j,i) = prim(IVX,k,js,i);
              prim(IVY,k,js-j,i) = prim(IVY,k,js,i);
              prim(IVZ,k,js-j,i) = prim(IVZ,k,js,i);
              prim(IPR,k,js-j,i) = prim(IPR,k,js,i);
            }
          }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
                  for(int k=ks; k<=ke; ++k){
                    for(int j=1; j<=(NGHOST); ++j){
                      for(int i=is; i<=ie+1; ++i){
                          b.x1f(k,js-j,i) = b.x1f(k,js,i);
                        }
                      }
                    }

                    for (int k=ks; k<=ke; ++k) {
                      for (int j=1; j<=(NGHOST); ++j) {
                        for (int i=is; i<=ie; ++i) {
                            b.x2f(k,js-j,i) = b.x2f(k,js,i);
                          }
                        }
                      }
                    

                      for (int k=ks; k<=ke+1; ++k) {
                        for (int j=1; j<=(NGHOST); ++j) {
                          for (int i=is; i<=ie; ++i) {
                              b.x3f(k,js-j,i) = b.x3f(k,js,i);
                            }
                          }
                        }
                  }       

      return;
      }
            
void MyBnd_ox2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

        for(int k=ks; k<=ke; ++k){
          for(int j=1; j<=(NGHOST); ++j){
            for(int i=is; i<=ie; ++i){
              prim(IDN,k,je+j,i) = prim(IDN,k,je,i) ;
              prim(IVX,k,je+j,i) = prim(IVX,k,je,i);
              prim(IVY,k,je+j,i) = prim(IVY,k,je,i);
              prim(IVZ,k,je+j,i) = prim(IVZ,k,je,i);
              prim(IPR,k,je+j,i) = prim(IPR,k,je,i);
            }
          }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
                  for(int k=ks; k<=ke; ++k){
                    for(int j=1; j<=(NGHOST); ++j){
                      for(int i=is; i<=ie+1; ++i){
                          b.x1f(k,je+j,i) = b.x1f(k,je,i);
                        }
                      }
                    }

                    for (int k=ks; k<=ke; ++k) {
                      for (int j=1; j<=(NGHOST); ++j) {
                        for (int i=is; i<=ie; ++i) {
                            b.x2f(k,je+j+1,i) = b.x2f(k,je+1,i);
                          }
                        }
                      }
                    

                      for (int k=ks; k<=ke+1; ++k) {
                        for (int j=1; j<=(NGHOST); ++j) {
                          for (int i=is; i<=ie; ++i) {
                              b.x3f(k,je+j,i) = b.x3f(k,je,i);
                            }
                          }
                        }
                  }         

        return;
      }

      

void MyBnd_ix3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

        for(int k=1; k<=(NGHOST); ++k){
          for(int j=js; j<=je; ++j){
            for(int i=is; i<=ie; ++i){
              prim(IDN,ks-k,j,i) = prim(IDN,ks,j,i);
              prim(IVX,ks-k,j,i) = prim(IVX,ks,j,i);
              prim(IVY,ks-k,j,i) = prim(IVY,ks,j,i);
              prim(IVZ,ks-k,j,i) = prim(IVZ,ks,j,i);
              prim(IPR,ks-k,j,i) = prim(IPR,ks,j,i) ;
            }
          }
        }

        if (MAGNETIC_FIELDS_ENABLED) {
                  for(int k=1; k<=(NGHOST); ++k){
                    for(int j=js; j<=je; ++j){
                      for(int i=is; i<=ie+1; ++i){
                          b.x1f(ks-k,j,i) = b.x1f(ks,j,i);
                        }
                      }
                    }

                    for (int k=1; k<=(NGHOST); ++k) {
                      for (int j=js; j<=je+1; ++j) {
                        for (int i=is; i<=ie; ++i) {
                            b.x2f(ks-k,j,i) = b.x2f(ks,j,i);
                          }
                        }
                      }
                    

                      for (int k=1; k<=(NGHOST); ++k) {
                        for (int j=js; j<=je; ++j) {
                          for (int i=is; i<=ie; ++i) {
                              b.x3f(ks-k,j,i) = b.x3f(ks,j,i);
                            }
                          }
                        }
                  }          

        return;
      }
              
void MyBnd_ox3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
               Real time, Real dt, int is, int ie, int js, int je, int ks, int ke ,int ngh){

      Real gamma = pmb->peos->GetGamma();

        for(int k=1; k<=(NGHOST); ++k){
          for(int j=js; j<=je; ++j){
            for(int i=is; i<=ie; ++i){
              prim(IDN,ke+k,j,i) = prim(IDN,ke,j,i);
              prim(IVX,ke+k,j,i) = prim(IVX,ke,j,i);
              prim(IVY,ke+k,j,i) = prim(IVY,ke,j,i);
              prim(IVZ,ke+k,j,i) = prim(IVZ,ke,j,i);
              prim(IPR,ke+k,j,i) = prim(IPR,ke,j,i);
            }
          }
        }

         if (MAGNETIC_FIELDS_ENABLED) {
                  for(int k=1; k<=(NGHOST); ++k){
                    for(int j=js; j<=je; ++j){
                      for(int i=is; i<=ie+1; ++i){
                          b.x1f(ke+k,j,i) = b.x1f(ke,j,i);
                        }
                      }
                    }

                    for (int k=1; k<=(NGHOST); ++k) {
                      for (int j=js; j<=je+1; ++j) {
                        for (int i=is; i<=ie; ++i) {
                            b.x2f(ke+k,j,i) = b.x2f(ke,j,i);
                          }
                        }
                      }
                    

                      for (int k=1; k<=(NGHOST); ++k) {
                        for (int j=js; j<=je; ++j) {
                          for (int i=is; i<=ie; ++i) {
                              b.x3f(ke+k+1,j,i) = b.x3f(ke+1,j,i);
                            }
                          }
                        }
                  }        

        return;
      }

void MeshBlock::UserWorkInLoop(void){
    Real x,y,z,press,rho,t,r,dv,phi,vx,vy,vz,r_vr,r_vphi,ang_z,
         bR,br,bphi,bz,bx,by,b,
         eta_ohm,eta_ad,reynolds_ohm,reynolds_ad,disk_radius=0.0;//変数定義

    Real rhomax = 0.0,tmax = 0.0,fcmass = 0.0,fcradius = 0.0,
         fcthickness = 0.0,of_imz = 0.0,of_vz = 0.0,of_mass=0.0,fcang_z=0.0,total_ang_z=0.0,
         Mf=0.0,bcz=0.0,br_ave=0.0,bR_ave=0.0,bphi_ave=0.0,b_ave=0.0,bz_ave = 0.0,
         eta_ohmmax=0.0,eta_admax=0.0;
         //出力したい変数に初期値＝0を代入(結果に関係なし)

    Real bcz_count=0,b_ave_count=0;

    if (pmy_mesh->ncycle % 100 == 0) { //100サイクルに一回出力  
        for (int k=ks; k<=ke; k++) {
            for (int j=js; j<=je; j++) {
                for (int i=is; i<=ie; i++) {
                    x = pcoord-> x1v(i);
                    y = pcoord-> x2v(j);
                    z = pcoord-> x3v(k);
                    press = phydro->w(IPR,k,j,i);
                    rho = phydro->w(IDN,k,j,i);
                    t = temp*press/rho;
                    dv = pcoord->GetCellVolume(k, j, i);
                    r = std::sqrt(x*x+y*y+z*z);//半径
                    phi = std::atan2(y,x);
                    eta_ohm = pfield->fdif.etaB(FieldDiffusion::DiffProcess::ohmic,k,j,i);
                    eta_ad = pfield->fdif.etaB(FieldDiffusion::DiffProcess::ambipolar,k,j,i);

                    vx = phydro->w(IM1,k,j,i);
                    vy = phydro->w(IM2,k,j,i);
                    vz = phydro->w(IM3,k,j,i);

                    //r*vr、r*vphi
                    r_vr = vx*x + vy*y + vz*z;
                    r_vphi = r*(-std::sin(phi)*vx + std::cos(phi)*vy);

                    ang_z = dv*(x*phydro->u(IM2,k,j,i)-y*phydro->u(IM1,k,j,i));//角運動量Lz
                    total_ang_z += ang_z;
                    
                    rhomax = std::max(rho,rhomax);

                    tmax = std::max(t,tmax); 

                    if (MAGNETIC_FIELDS_ENABLED) {
                        bx = pfield->bcc(IB1,k,j,i);
                        by = pfield->bcc(IB2,k,j,i);
                        bz = pfield->bcc(IB3,k,j,i);

                        b = std::sqrt(bx*bx+by*by+bz*bz);//bccの大きさ

                        bR = std::cos(phi)*bx + std::sin(phi)*by;//円筒座標
                        br = (bx*x + by*y + bz*z)/(r+1e-16);//球座標
                        bphi = -std::sin(phi)*bx + std::cos(phi)*by;
                    }

                    //コア内
                    if((rho > 1e-13/rho0 && std::abs(r_vr) < std::abs(r_vphi) && r_vr < 0.0) || rho > rhocrit){//(落下速度<回転速度 or 1e-12 > rho) and 1e-13 > rho
                        
                        fcmass += dv*rho;//コアの質量

                        fcang_z += ang_z;//コアの角運動量

                        fcradius = std::max(r,fcradius);//コアの半径
                        
                        if (std::abs(x) < pcoord->dx1f(i) && std::abs(y) < pcoord->dx2f(j)){//z軸
                            fcthickness = std::max(std::abs(z),fcthickness);
                        }

                        if (MAGNETIC_FIELDS_ENABLED) {
                            br_ave += br;
                            bR_ave += bR;
                            bphi_ave += bphi;
                            b_ave += b;
                            bz_ave += bz;

                            b_ave_count += 1;

                            if (std::abs(z) < pcoord->dx3f(k) && z > 0.0){//xy平面
                                Mf += bz * pcoord->GetFace3Area(k,j,i);
                            }
                        }
                    }

                    if (std::abs(x) < pcoord->dx1f(i) && std::abs(y) < pcoord->dx2f(j) && std::abs(z) < pcoord->dx3f(k)){//中心
                                bcz += bz;//平均をとる

                                //磁気抵抗率について調べる cloudの中心の値をとってくる
                                eta_ohmmax += eta_ohm;
                                eta_admax += eta_ad;

                                bcz_count = 1;
                    }

                    //アウトフローについて調べる。vzとzが同じ符号の領域をアウトフローの領域とする。
                    if(vz*z > 0.0){
                        of_mass += rho*dv;
                        of_vz = std::max(of_vz,std::abs(vz));
                        of_imz = std::max(of_imz,std::abs(rho*vz));
                    }
                }
            }        
        }    
        //maxをとるもの
        ruser_meshblock_data[0](0) = rhomax;//最大密度
        ruser_meshblock_data[0](1) = tmax;//最大温度
        ruser_meshblock_data[0](2) = fcradius;//xy平面での半径
        ruser_meshblock_data[0](3) = fcthickness;//z軸の大きさ
        ruser_meshblock_data[0](4) = of_vz;//アウトフローの最大vz
        ruser_meshblock_data[0](5) = of_imz;//アウトフローの最大imz
        //sumをとるもの
        ruser_meshblock_data[1](0) = fcmass;//コアの質量
        ruser_meshblock_data[1](1) = total_ang_z;//全体のLz
        ruser_meshblock_data[1](2) = fcang_z;//回転している領域のLz
        ruser_meshblock_data[1](3) = bcz/bcz_count;//中心の磁場強度
        ruser_meshblock_data[1](4) = b_ave/b_ave_count;///コア内の平均磁場強度
        ruser_meshblock_data[1](5) = br_ave/b_ave_count;///コア内のr方向平均磁場強度
        ruser_meshblock_data[1](6) = bR_ave/b_ave_count;///コア内のR方向平均磁場強度
        ruser_meshblock_data[1](7) = bphi_ave/b_ave_count;///コア内のphi方向平均磁場強度
        ruser_meshblock_data[1](8) = bz_ave/b_ave_count;///コア内のz方向平均磁場強度
        ruser_meshblock_data[1](9) = Mf;///コア内のxy平面での磁束
        ruser_meshblock_data[1](10) = of_mass;//アウトフローのmass
        ruser_meshblock_data[1](11) = eta_ohmmax/bcz_count;//中心のオーム抵抗率etaohm
        ruser_meshblock_data[1](12) = eta_admax/bcz_count;//中心のAD抵抗率etaad

    }

    return;
}

void Mesh::UserWorkInLoop(void) {
  //MPIが有効ならプログラムに入る
    #ifdef MPI_PARALLEL
        MPI_Allreduce(MPI_IN_PLACE, my_blocks(0)->ruser_meshblock_data[0].data(), 6, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, my_blocks(0)->ruser_meshblock_data[1].data(), 13, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    #endif

    Real rhomax = 0.0, tmax = 0.0, fcmass = 0.0, fcradius = 0.0, fcthickness = 0.0
         ,total_ang_z = 0.0,fcang_z=0.0,of_imz = 0.0,of_vz = 0.0,of_mass=0.0,
         Mf=0.0,bcz=0.0,br_ave=0.0,bR_ave=0.0,bphi_ave=0.0,b_ave=0.0,bz_ave = 0.0,
         eta_ohm=0.0,eta_ad=0.0,reynolds_ohm=0.0,reynolds_ad=0.0,disk_radius=0.0;
    Real bcz_count=0,b_ave_count=0;

    if (ncycle % 100 == 0) {
        for (int n =0; n < nblocal; ++n) {
        rhomax = std::max(rhomax, my_blocks(n)->ruser_meshblock_data[0](0));

        tmax = std::max(tmax, my_blocks(n)->ruser_meshblock_data[0](1));

        fcmass += my_blocks(n)->ruser_meshblock_data[1](0);

        fcradius= std::max(fcradius, my_blocks(n)->ruser_meshblock_data[0](2));

        fcthickness = std::max(fcthickness, my_blocks(n)->ruser_meshblock_data[0](3));

        total_ang_z += my_blocks(n)->ruser_meshblock_data[1](1);

        fcang_z += my_blocks(n)->ruser_meshblock_data[1](2);
        
        if(std::abs(my_blocks(n)->ruser_meshblock_data[1](4)) > 1e-15){
            b_ave += my_blocks(n)->ruser_meshblock_data[1](4);
            
            br_ave += my_blocks(n)->ruser_meshblock_data[1](5);
            
            bR_ave += my_blocks(n)->ruser_meshblock_data[1](6);
            
            bphi_ave += my_blocks(n)->ruser_meshblock_data[1](7);
            
            bz_ave += my_blocks(n)->ruser_meshblock_data[1](8);

            b_ave_count += 1;
        }

        Mf += my_blocks(n)->ruser_meshblock_data[1](9);

        of_mass += my_blocks(n)->ruser_meshblock_data[1](10);

        of_vz = std::max(of_vz,my_blocks(n)->ruser_meshblock_data[0](4));

        of_imz = std::max(of_imz,my_blocks(n)->ruser_meshblock_data[0](5));

        if(std::abs(my_blocks(n)->ruser_meshblock_data[1](7)) > 1e-15){
          bcz += my_blocks(n)->ruser_meshblock_data[1](7);
          bcz_count+=1;

          eta_ohm += my_blocks(n)->ruser_meshblock_data[1](11);

          eta_ad += my_blocks(n)->ruser_meshblock_data[1](12);

        }
        }

        FILE *fp1,*fp2;

        if (Globals::my_rank==0) {
        fp1 = fopen("data.dat","a");
        if (fp1==NULL){
            printf("Cannot open data.dat\n");
            exit(1);
        }
        //時間、最大密度、最大温度、コアの質量、コアの半径、z方向の厚み、全体のLz、回転している領域のLz
        fprintf(fp1,"%e %e %e %e %e %e %e %e %e\n", time, rhomax, tmax, fcmass, fcradius,fcthickness,total_ang_z,fcang_z,disk_radius);
        fclose(fp1);

        fp2 = fopen("data_bcc.dat","a");
        if (fp2==NULL){
            printf("Cannot open data.dat\n");
            exit(1);
        }
        //時間、最大密度、中心の磁場強度、コアの平均磁場強度、コア内のr方向平均磁場強度、コア内のR方向平均磁場強度
        //コア内のphi方向平均磁場強度、コア内のz方向平均磁場強度
        //コア内のxy平面での磁束、アウトフローのmass、アウトフローの最大vz、アウトフローの最大imz=rho*vz、
        //中心のohm磁気抵抗率、中心のAD磁気抵抗率、中心のohm磁気レイノルズ数、中心のAD磁気レイノルズ数
        fprintf(fp2,"%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n", 
        time, rhomax, bcz/bcz_count,b_ave/b_ave_count,
        br_ave/b_ave_count,bR_ave/b_ave_count,
        bphi_ave/b_ave_count,bz_ave/b_ave_count,
        Mf,of_mass,of_vz,of_imz,
        eta_ohm/bcz_count,eta_ad/bcz_count,reynolds_ohm/bcz_count,reynolds_ad/bcz_count);
        fclose(fp2);
        }
    }
    return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin){
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      for(int i=is; i<=ie; i++) {
        user_out_var(0,k,j,i) = pfield->fdif.etaB(FieldDiffusion::DiffProcess::ohmic,k,j,i);
        user_out_var(1,k,j,i) = pfield->fdif.etaB(FieldDiffusion::DiffProcess::ambipolar,k,j,i);
      }
    }
  }
}
