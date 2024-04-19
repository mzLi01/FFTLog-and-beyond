enum BesselIntType{
	J_ELL,
	J_ELL_DERIV,
	J_ELL_DERIV_2,
	J_ELL_SQUARED,
	J_ELL1_J_ELL2_EQ,
	J_ELL1_J_ELL2_NEQ,
	J_LP1_J_LM1_EQ,
};
typedef struct config
{
	double nu;
	double c_window_width;
	enum BesselIntType type;
	long N_pad;
} config;

void cfftlog_ells(double *x, double *fx, long N, double y_ratio, config *config, double* ell1, double *ell2, long Nell, double **y, double **Fy);

void cfftlog_ells_increment(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy);

void cfftlog_ells_wrapper(double *x, double *fx, long N, double y_ratio, double* ell1, double *ell2, long Nell, double **y, double **Fy, double nu, double c_window_width, enum BesselIntType type, long N_pad);

void cfftlog_modified_ells(double *x, double *fx, long N, config *config, double* ell, long Nell, double **y, double **Fy);

void cfftlog_modified_ells_wrapper(double *x, double *fx, long N, double* ell, long Nell, double **y, double **Fy, double nu, double c_window_width, int derivative, int j_squared, long N_pad);
