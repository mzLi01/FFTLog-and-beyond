#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <gsl/gsl_sf_gamma.h>
#include <flint/acb.h>
#include <flint/arb.h>
#include <flint/arf.h>
#include <flint/acb_hypgeom.h>

#include "utils.h"
#include "utils_complex.h"


double complex lngamma_gsl(double complex z) {
	gsl_sf_result lnr, arg;
	int status;
	status = gsl_sf_lngamma_complex_e(creal(z), cimag(z), &lnr, &arg);
	if(status) {
		printf("Error in gsl_sf_lngamma_complex_e: %d\n", status);
		exit(1);
	}
	return lnr.val + I*arg.val;
}

double complex acb_to_complex(acb_t z) {
	double complex result;
	result = arf_get_d(arb_midref(acb_realref(z)), ARF_RND_NEAR) + I * arf_get_d(arb_midref(acb_imagref(z)), ARF_RND_NEAR);
	return result;
}

short arb_precise(arb_t x) {
	/* if error/value>1, accr_bits<0;
	if value=nan, accr_bits=1*/
	return (arb_rel_accuracy_bits(x) > 1);
}

void acb_hypgeom_2F1_renorm_adapt_prec(acb_t result, acb_t a, acb_t b, acb_t c, acb_t x, slong *prec) {
	while (1) {
		acb_hypgeom_2f1(result, a, b, c, x, 1, *prec);
		if (arb_precise(acb_realref(result)) && arb_precise(acb_imagref(result))) {
			return;
		}
		*prec *= 2;
	}
}

double complex hyperg_2F1_renorm_arb(double complex a, double complex b, double complex c, double complex x, slong *prec) {
	acb_t a_acb, b_acb, c_acb, x_acb, result_acb;
	double complex result;
	acb_init(a_acb); acb_init(b_acb);
	acb_init(c_acb); acb_init(x_acb);
	acb_init(result_acb);
	acb_set_d_d(a_acb, creal(a), cimag(a)); acb_set_d_d(b_acb, creal(b), cimag(b));
	acb_set_d_d(c_acb, creal(c), cimag(c)); acb_set_d_d(x_acb, creal(x), cimag(x));

	acb_hypgeom_2F1_renorm_adapt_prec(result_acb, a_acb, b_acb, c_acb, x_acb, prec);
	result = acb_to_complex(result_acb);

	acb_clear(a_acb); acb_clear(b_acb); acb_clear(c_acb); acb_clear(x_acb); acb_clear(result_acb);
	return result;
}

double complex ln_g_m_vals(double mu, double complex q) {
/* similar routine as python version.
use asymptotic expansion for large |mu+q| */
	double complex asym_plus = (mu+1+ q)/2.;
	double complex asym_minus= (mu+1- q)/2.;

	return ln_g_m_ratio(asym_plus, asym_minus);
}

double complex ln_g_m_ratio(double complex a, double complex b) {
/* ln(gamma(a)/gamma(b))
use asymptotic expansion for large |a| */
	return (a-0.5)*clog(a) - (b-0.5)*clog(b) - a + b \
		+1./12 *(1./a - 1./b) \
		+1./360.*(1./cpow(b,3) - 1./cpow(a,3)) \
		+1./1260*(1./cpow(a,5) - 1./cpow(b,5));
}

void g_l(double l, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
Calculate g_l = exp( zln2 + lngamma( (l+nu)/2 + I*eta/2 ) - lngamma( (3+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for (i = 0; i < N; i++)
	{
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			gl[i] = cexp(z*log(2.) + lngamma_gsl((l+z)/2.) - lngamma_gsl((3.+l-z)/2.));
		}else{
			gl[i] = cexp(z*log(2.) + ln_g_m_vals(l+0.5, z-1.5));
		}
	}
}

void g_l_1(double l, double nu, double *eta, double complex *gl1, long N) {
/* z = nu + I*eta
Calculate g_l_1 = exp(zln2 + lngamma( (l+nu-1)/2 + I*eta/2 ) - lngamma( (4+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + lngamma_gsl((l+z-1.)/2.) - lngamma_gsl((4.+l-z)/2.));
		}else{
			gl1[i] = -(z-1.)* cexp((z-1.)*log(2.) + ln_g_m_vals(l+0.5, z-2.5));
		}
	}
}

void g_l_2(double l, double nu, double *eta, double complex *gl2, long N) {
/* z = nu + I*eta
Calculate g_l_1 = exp(zln2 + lngamma( (l+nu-2)/2 + I*eta/2 ) - lngamma( (5+l-nu)/2 - I*eta/2 ) ) */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			gl2[i] = (z-1.)* (z-2.)* cexp((z-2.)*log(2.) + lngamma_gsl((l+z-2.)/2.) - lngamma_gsl((5.+l-z)/2.));
		}else{
			gl2[i] = (z-1.)* (z-2.)* cexp((z-2.)*log(2.) + ln_g_m_vals(l+0.5, z-3.5));
		}
	}
}

void h_l(double l, double nu, double *eta, double complex *hl, long N) {
/* z = nu + I*eta */
	long i;
	double complex z;
	for(i=0; i<N; i++) {
		z = nu+I*eta[i];
		if(l+fabs(eta[i])<200){
			hl[i] = cexp(lngamma_gsl(l +z/2.) + lngamma_gsl((2.-z)/2.) \
					- lngamma_gsl(2.+l-z/2.) - lngamma_gsl((3.-z)/2.) );
		}else{
			hl[i] = cexp(ln_g_m_vals(2*l+1., z-2.) + ln_g_m_ratio((2.-z)/2., (3.-z)/2.));
		}
	}
}

void g_l1_l2_unequal(double l1, double l2, double y_ratio, double nu, double *eta, double complex *gl, long N) {
	/* z = nu + I*eta
	g_{l1,l2} = 4/sqrt(pi) \int dx/x j_l1(x) j_l2(beta*x) x^{nu+i*eta},
	where beta is y_ratio, required beta<1 */
	slong prec = 64;
	double complex z, half_z, part1;
	acb_t hyperg_a, hyperg_b, hyperg_c, hyperg_x, hyperg_result;
	double half_l1pl2 = (l1 + l2) / 2., half_l1ml2 = (l1 - l2) / 2.;
	acb_init(hyperg_a); acb_init(hyperg_b); acb_init(hyperg_c); acb_init(hyperg_x); acb_init(hyperg_result);
	acb_set_d(hyperg_a, nu/2. - half_l1ml2 - 0.5);
	acb_set_d(hyperg_b, nu/2. + half_l1pl2);
	acb_set_d(hyperg_c, l2 + 1.5);
	acb_set_d(hyperg_x, y_ratio * y_ratio);

	for (long i = 0; i < N; i++)
	{
		z = nu + I * eta[i];
		half_z = z / 2.;
		arb_set_d(acb_imagref(hyperg_a), cimag(half_z));
		arb_set_d(acb_imagref(hyperg_b), cimag(half_z));

		part1 = sqrt(M_PI) * cexp(
			(z - 1.) * log(2.) + l2 * log(y_ratio) +\
			lngamma_gsl(half_l1pl2 + half_z) - lngamma_gsl(half_l1ml2 - half_z + 1.5));
		acb_hypgeom_2F1_renorm_adapt_prec(hyperg_result, hyperg_a, hyperg_b, hyperg_c, hyperg_x, &prec);
		gl[i] = part1 * acb_to_complex(hyperg_result);
	}
	acb_clear(hyperg_a); acb_clear(hyperg_b); acb_clear(hyperg_c); acb_clear(hyperg_x); acb_clear(hyperg_result);
}

void g_l1_l2_neq_wrapper(double l1, double l2, double y_ratio, double nu, double *eta, double *gl_re, double *gl_im, long N)
{
	double complex *gl;
	gl = malloc(N * sizeof(double complex));
	g_l1_l2_unequal(l1, l2, y_ratio, nu, eta, gl, N);
	for (long i = 0; i < N; i++)
	{
		gl_re[i] = creal(gl[i]);
		gl_im[i] = cimag(gl[i]);
	}
	free(gl);
}

void g_l1_l2_unequal_2(double l1, double l2, double y_ratio, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
   g_{l1,l2} = 4/sqrt(pi) \int dx/x j_l1(x) j_l2(beta*x) x^{nu+i*eta},
   where beta is y_ratio, required beta<1 */
	slong prec = 64;
	double complex z, half_z;
	double beta2 = y_ratio * y_ratio, half_l1pl2 = (l1 + l2) / 2., half_l1ml2 = (l1 - l2) / 2.;
	for (long i = 0; i < N; i++)
	{
		z = nu+I*eta[i];
		half_z = z/2.;
		gl[i] = sqrt(M_PI) * cexp(
			(z-1.)*log(2.) + l2*log(y_ratio) + lngamma_gsl(half_l1pl2+half_z) - lngamma_gsl(half_l1ml2-half_z+1.5)) *\
			hyperg_2F1_renorm_arb(half_z-half_l1ml2-0.5, half_z+half_l1pl2, l2+1.5, beta2, &prec);
	}
}

void g_l1_l2_equal(double l1, double l2, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
   g_{l1,l2} = 4/sqrt(pi) \int dx/x j_l1(x) j_l2(x) x^{nu+i*eta} */
	long i;
	double complex z, half_z;
	double half_l1pl2=(l1+l2)/2., half_l1ml2=(l1-l2)/2.;
	for (i = 0; i < N; i++)
	{
		z = nu+I*eta[i];
		half_z = z/2.;
		gl[i] = sqrt(M_PI) * cexp(
			(z-1.)*log(2.) + lngamma_gsl(2.-z) + lngamma_gsl(half_l1pl2+half_z) -\
			lngamma_gsl(half_l1ml2-half_z+1.5) - lngamma_gsl(-half_l1ml2-half_z+1.5) - lngamma_gsl(half_l1pl2-half_z+2.));
	}
}

void g_lpm1_equal(double l, double nu, double *eta, double complex *gl, long N) {
/* z = nu + I*eta
   g_{l+1,l-1} = 4/sqrt(pi) \int dx/x j_{l+1}(x) j_{l-1}(x) x^{nu+i*eta} */
	long i;
	double complex z, half_z;
	for (i = 0; i < N; i++)
	{
		z = nu+I*eta[i];
		half_z = z/2.;
		gl[i] = 1. / (1.5-half_z) * cexp(
			lngamma_gsl(l+half_z) + lngamma_gsl(1.-half_z) -\
			lngamma_gsl(0.5-half_z) - lngamma_gsl(2.+l-half_z));
	}
}

void g_l_modified(double l, double nu, double *eta, double complex *gl, long N) {
	g_l(l, nu - 2, eta, gl, N);
}
void g_l_1_modified(double l, double nu, double *eta, double complex *gl, long N) {
	g_l_1(l, nu - 2, eta, gl, N);
}
void g_l_2_modified(double l, double nu, double *eta, double complex *gl, long N) {
	g_l_2(l, nu - 2, eta, gl, N);
}

void c_window(double complex *out, double c_window_width, long halfN) {
	// 'out' is (halfN+1) complex array
	long Ncut;
	Ncut = (long)(halfN * c_window_width);
	long i;
	double W;
	for(i=0; i<=Ncut; i++) { // window for right-side
		W = (double)(i)/Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI/Ncut);
		out[halfN-i] *= W;
	}
}

