#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <gsl/gsl_sf_bessel.h>

#define gamma_max (1000.)
#define MINW  1.e-12
#define MAXW  1.e15
#define MINT  0.0001
#define MAXT  1.e4
#define NW  220
#define NT  80

#define SIGMA_THOMSON 0.665245873e-24

#define MAXGAMMA  12.
#define DMUE    0.005
#define DGAMMAE   0.005

double hc_klein_nishina(double w);
double dNdgammae(double thetae, double gammae);
double boostcross(double k, double p, double ke);

double total_compton_cross_num(double w, double thetae, double norm)
{
  double dmue, dgammae, mue, gammae, f, cross;

  // check for easy-to-do limits
  if (thetae < MINT && w < MINW) {
    return SIGMA_THOMSON;
  }

  if (thetae < MINT) {
    return hc_klein_nishina(w) * SIGMA_THOMSON;
  }
  dmue = DMUE;
  dgammae = thetae * DGAMMAE;

  // integrate over mu_e, gamma_e, where mu_e is the cosine of the
  // angle between k and u_e, and the angle k is assumed to lie,
  // wlog, along the z axis
  cross = 0.;
  for (mue = -1. + dmue/2.; mue < 1.; mue += dmue)
    for (gammae= 1. + dgammae/2; gammae < 1. + MAXGAMMA*thetae; gammae += dgammae) {

      f = 0.5 * norm*dNdgammae(thetae, gammae);
      // fprintf(stderr,"dndgammae(%.14g,%.14g) = %.14g\n",gammae,thetae,dNdgammae(thetae, gammae, rpars));
      cross += dmue * dgammae * boostcross(w, mue, gammae) * f;

    }
  // fprintf(stderr,"compton cross params: %g %g \n result: %g \n",w,thetae,cross*SIGMA_THOMSON);
  fprintf(stderr,"%.5g\n",cross*SIGMA_THOMSON);

  return cross * SIGMA_THOMSON;
}


double dNdgammae(double thetae, double gammae)
{

  // multiply K2(1/Thetae) by e^(1/Thetae) for numerical purposes
  double K2f;
  if (thetae > 1.e-2) {
    K2f = gsl_sf_bessel_Kn(2, 1. / thetae) * exp(1. / thetae);
  } else {
    K2f = sqrt(M_PI * thetae / 2.);
  }

  return (gammae * sqrt(gammae * gammae - 1.) / (thetae * K2f)) *
    exp(-(gammae - 1.) / thetae);
}

double boostcross(double w, double mue, double gammae)
{
  double we, boostcross, v;

  // energy in electron rest frame 
  v = sqrt(gammae * gammae - 1.) / gammae;
  we = w * gammae * (1. - mue * v);
  boostcross = hc_klein_nishina(we) * (1. - mue * v);

  if (boostcross > 2) {
    fprintf(stderr, "w,mue,gammae: %g %g %g\n", w, mue,
      gammae);
    fprintf(stderr, "v,we, boostcross: %g %g %g\n", v, we,
      boostcross);
    fprintf(stderr, "kn: %g %g %g\n", v, we, boostcross);
  }

  if (isnan(boostcross)) {
    fprintf(stderr, "isnan: %g %g %g\n", w, mue, gammae);
    exit(0);
  }

  return boostcross;
}

double hc_klein_nishina(double we)
{
  double sigma;

  if (we < 1.e-3)
    return 1. - 2. * we;

  sigma = (3. / 4.) * (2. / (we * we) +
           (1. / (2. * we) -
            (1. + we) / (we * we * we)) * log(1. + 2. * we) +
           (1. + we) / ((1. + 2. * we) * (1. + 2. * we))
      );

  return sigma;
}

int main(){
    // fprintf(stderr,"\ntesting grmonty hotcross\n");
    double w = 1e-12;
    double thetae = 1e-4;
    double norm = 1;

    double kev_to_cgs = 0.0019569511835738733;
    double temps[6] = {1,10,100,500,1000,5000};
    double photon_energies[6] = {1,10,100,500,1000,5000};
    for (int i=0;i<6;i++){
      temps[i] *=kev_to_cgs;
      photon_energies[i]*=kev_to_cgs;
    }
    for (int i =0;i<6;i++) for(int j=0;j<6;j++) total_compton_cross_num(photon_energies[i], temps[j], norm);
    return 0;
}