#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include <time.h>

#include <fftw3.h>

#include "utils.h"
#include "utils_complex.h"
#include "cfftlog.h"


void g_l_choose(double l1, double l2, double y_ratio, double *eta_m, config *config, double complex *gl, long N) {
	switch(config->type) {
		case J_ELL: 
			g_l(l1, config->nu, eta_m, gl, N);
			break;
		case J_ELL_DERIV: 
			g_l_1(l1, config->nu, eta_m, gl, N);
			break;
		case J_ELL_DERIV_2: 
			g_l_2(l1, config->nu, eta_m, gl, N);
			break;
		case J_ELL_SQUARED:
			h_l(l1, config->nu, eta_m, gl, N);
			break;
		case J_ELL1_J_ELL2_EQ:
			g_l1_l2_equal(l1, l2, config->nu, eta_m, gl, N);
			break;
		case J_ELL1_J_ELL2_NEQ:
			g_l1_l2_unequal(l1, l2, y_ratio, config->nu, eta_m, gl, N);
			break;
		case J_LP1_J_LM1_EQ:
			g_lpm1_equal(l1, config->nu, eta_m, gl, N);
			break;
		default:
			printf("Integral Not Supported!\n");
			exit(0);
		}
}

void cfftlog_ells_wrapper(double *x, double *fx, long N, double y_ratio, double* ell1, double *ell2, long Nell, double **y, double **Fy, double nu, double c_window_width, enum BesselIntType type, long N_pad){
	config my_config;
	my_config.nu = nu;
	my_config.c_window_width = c_window_width;
	my_config.type = type;
	my_config.N_pad = N_pad;
	cfftlog_ells(x, fx, N, y_ratio, &my_config, ell1, ell2, Nell, y, Fy);
}

void cfftlog_ells(double *x, double *fx, long N, double y_ratio, config *config, double* ell1, double *ell2, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		g_l_choose(ell1[j], ell2[j], y_ratio, eta_m, config, gl, halfN+1);
		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell1[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i]) ;
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] = out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}

/*
void cfftlog_modified_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, int j_squared, long N_pad){
	config my_config;
	my_config.nu = nu;
	my_config.c_window_width = c_window_width;
	my_config.derivative = derivative;
	my_config.j_squared = j_squared;
	my_config.N_pad = N_pad;
	cfftlog_modified_ells(x, fx, N, &my_config, ell, Nell, y, Fy);
}
// Only modification is g_l functions, when the integrand function is f(x)/(xy)^2
void cfftlog_modified_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		if(config->j_squared ==0){
			switch(config->derivative) {
				case 0: g_l_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
				case 1: g_l_1_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
				case 2: g_l_2_modified(ell[j], config->nu, eta_m, gl, halfN+1); break;
				default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
			}
		}else{
			printf("j_sqr Integral Not Supported for modified version!\n");
			exit(1);
		}


		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i] ) ;
			// printf("gl:%e\n", gl[i]);
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] = out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}


void cfftlog_ells_increment(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out, *out_vary;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out_vary, out_ifft, FFTW_ESTIMATE);

	for(j=0; j<Nell; j++){
		if(config->j_squared ==0){
			switch(config->derivative) {
				case 0: g_l(ell[j], config->nu, eta_m, gl, halfN+1); break;
				case 1: g_l_1(ell[j], config->nu, eta_m, gl, halfN+1); break;
				case 2: g_l_2(ell[j], config->nu, eta_m, gl, halfN+1); break;
				default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
			}
		}else{
			h_l(ell[j], config->nu, eta_m, gl, halfN+1);
		}

		// calculate y arrays
		for(i=0; i<N_original; i++) {y[j][i] = (ell[j]+1.) / x[N_original-1-i];}
		y0 = y[j][0];

		for(i=0; i<=halfN; i++) {
			out_vary[i] = conj(out[i] * cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i]) ;
			// printf("gl:%e\n", gl[i]);
		}

		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++) {
			Fy[j][i] += out_ifft[i+N_pad] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}
	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
}
*/