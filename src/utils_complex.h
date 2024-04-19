#include <complex.h>
#include <fftw3.h>

void g_l(double l, double nu, double *eta, double complex *gl, long N);
void g_l_1(double l, double nu, double *eta, double complex *gl1, long N);
void g_l_2(double l, double nu, double *eta, double complex *gl2, long N);
void g_l1_l2_unequal(double l1, double l2, double y_ratio, double nu, double *eta, double complex *gl, long N);
void g_l1_l2_neq_wrapper(double l1, double l2, double y_ratio, double nu, double *eta, double *gl_re, double *gl_im, long N);
double complex hyperg_2F1_renorm_arb(double complex a, double complex b, double complex c, double complex x, long *prec);

void g_l1_l2_equal(double l1, double l2, double nu, double *eta, double complex *gl, long N);
void g_lpm1_equal(double l, double nu, double *eta, double complex *gl, long N);

void g_l_modified(double l, double nu, double *eta, double complex *gl, long N);
void g_l_1_modified(double l, double nu, double *eta, double complex *gl1, long N);
void g_l_2_modified(double l, double nu, double *eta, double complex *gl2, long N);

void h_l(double l, double nu, double *eta, double complex *hl, long N);


void c_window(double complex *out, double c_window_width, long halfN);

double complex ln_g_m_vals(double mu, double complex q);
double complex ln_g_m_ratio(double complex a, double complex b);