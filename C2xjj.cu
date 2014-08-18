/*
 * Two Underdamped Brownian Particles
 *
 */

#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265358979f

//model
__constant__ float d_amp, d_omega, d_force, d_gam, d_Dg;//, d_Dp, d_lambda, d_mean, d_fa, d_fb, d_mua, d_mub;
//__constant__ int d_comp;
float h_omega;//, h_lambda, h_fa, h_fb, h_mua, h_mub, h_mean;
//int h_comp;

//simulation
int h_dev, h_block, h_grid, h_spp, h_samples, h_2ndorder, h_trigger, h_paths, h_periods, h_trans;//, h_initnoise;
long h_threads, h_steps;
__constant__ int d_spp, d_2ndorder, d_samples, d_trigger, d_paths;//, d_initnoise;

//output
char *h_domain;
char h_domainx, h_domainy;
float h_beginx, h_endx, h_beginy, h_endy;
int h_logx, h_logy, h_points, h_moments, h_traj, h_hist;
__constant__ char d_domainx;
__constant__ int d_moments, d_points;

//vector
float *h_x1, *h_fx1, *h_v1, *h_x2, *h_fx2, *h_v2, *h_w, *h_fw, *h_sv1, *h_sv12, *h_sv2, *h_sv22, *h_dx;
float *d_x1, *d_fx1, *d_v1, *d_x2, *d_fx2, *d_v2, *d_w, *d_fw, *d_sv1, *d_sv12, *d_sv2, *d_sv22, *d_dx;
//int *d_pcd, *d_dcd, *d_dst;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_i, size_ui, size_p;
curandGenerator_t gen;

//command line arguments
static struct option options[] = {
    {"amp", required_argument, NULL, 'a'},
    {"omega", required_argument, NULL, 'b'},
    {"force", required_argument, NULL, 'c'},
    {"gam", required_argument, NULL, 'd'},
    {"Dg", required_argument, NULL, 'e'},
/*    {"Dp", required_argument, NULL, 'f'},
    {"lambda", required_argument, NULL, 'g'},
    {"comp", required_argument, NULL, 'h'},*/
    {"dev", required_argument, NULL, 'i'},
    {"block", required_argument, NULL, 'j'},
    {"paths", required_argument, NULL, 'k'},
    {"periods", required_argument, NULL, 'l'},
    {"trans", required_argument, NULL, 'm'},
    {"spp", required_argument, NULL, 'n'},
    {"samples", required_argument, NULL, 'o'},
    {"algorithm", required_argument, NULL, 'p'},
    {"mode", required_argument, NULL, 'q'},
    {"domain", required_argument, NULL, 'r'},
    {"domainx", required_argument, NULL, 's'},
    {"domainy", required_argument, NULL, 't'},
    {"logx", required_argument, NULL, 'u'},
    {"logy", required_argument, NULL, 'v'},
    {"points", required_argument, NULL, 'w'},
    {"beginx", required_argument, NULL, 'y'},
    {"endx", required_argument, NULL, 'z'},
    {"beginy", required_argument, NULL, 'A'},
    {"endy", required_argument, NULL, 'B'}
/*    {"mean", required_argument, NULL, 'C'},
    {"fa", required_argument, NULL, 'D'},
    {"fb", required_argument, NULL, 'E'},
    {"mua", required_argument, NULL, 'F'},
    {"mub", required_argument, NULL, 'G'}*/
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --amp=FLOAT         set the harmonic driving amplitude 'a' to FLOAT\n");
    printf("    -b, --omega=FLOAT       set the harmonic driving frequency '\\omega' to FLOAT\n");
    printf("    -c, --force=FLOAT       set the external bias 'F' to FLOAT\n");
    printf("    -d, --gam=FLOAT         set the viscosity '\\gamma' to FLOAT\n");
    printf("    -e, --Dg=FLOAT          set the Gaussian noise intensity 'D_G' to FLOAT\n");
/*    printf("    -f, --Dp=FLOAT          set the Poissonian noise intensity 'D_P' to FLOAT\n");
    printf("    -g, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n");
    printf("    -h, --comp=INT          choose between biased and unbiased Poissonian or dichotomous noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("    -D, --fa=FLOAT          set the first state of the dichotomous noise 'F_a' to FLOAT\n");
    printf("    -E, --fb=FLOAT          set the second state of the dichotomous noise 'F_b' to FLOAT\n");
    printf("    -F, --mua=FLOAT         set the transition rate of the first state of dichotomous noise '\\mu_a' to FLOAT\n");
    printf("    -G, --mub=FLOAT         set the transition rate of the second state of dichotomous noise '\\mu_b' to FLOAT\n");
    printf("    -C, --mean=FLOAT        if is nonzero, fix the mean value of Poissonian noise or dichotomous noise to FLOAT, matters only for domains p, l, i, j, m or n\n");*/
    printf("Simulation params:\n");
    printf("    -i, --dev=INT           set the gpu device to INT\n");
    printf("    -j, --block=INT         set the gpu block size to INT\n");
    printf("    -k, --paths=INT         set the number of paths to INT\n");
    printf("    -l, --periods=INT       set the number of periods to INT\n");
    printf("    -m, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -n, --spp=INT           specify how many integration steps should be calculated\n");
    printf("                            for a single period of the driving force\n");
    printf("    -o, --samples=INT       specify how many integration steps should be calculated for a single kernel call\n");
    printf("    -p, --algorithm=STRING  sets the algorithm. STRING can be one of:\n");
    printf("                            predcorr: simplified weak order 2.0 adapted predictor-corrector\n");
    printf("                            euler: simplified weak order 1.0 regular euler-maruyama\n");
    printf("Output params:\n");
    printf("    -q, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first two moments <<v>>, <<v^2>> and diffusion coefficient\n");
    printf("                            trajectory: ensemble averaged <x>(t), <v>(t) and <x^2>(t), <v^2>(t)\n");
    printf("                            histogram: the final position x and velocity v of all paths\n");
    printf("    -r, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter; 2d: two parameters at once\n");
    printf("    -s, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            a: amp; w: omega, f: force; g: gam; D: Dg\n");// p: Dp; l: lambda; i: fa; j: fb; m: mua; n: mub\n");
    printf("    -t, --domainy=CHAR      sets the second domain of the moments (only if --domain=2d). CHAR can be the same as above.\n");
    printf("    -u, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -v, --logy=INT          the same as above but for domainy\n");
    printf("    -w, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -y, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -z, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("    -A, --beginy=FLOAT      the same as --beginx, but for domainy\n");
    printf("    -B, --endy=FLOAT        the same as --endx, but for domainy\n");
    printf("\n");
}

//parse command line arguments
void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

//    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:y:z:A:B:C:D:E:F:G", options, NULL)) != EOF) {
    while( (c = getopt_long(argc, argv, "a:b:c:d:e:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:y:z:A:B", options, NULL)) != EOF) {
        switch (c) {
            case 'a':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_amp, &ftmp, sizeof(float));
                break;
            case 'b':
                h_omega = atof(optarg);
                cudaMemcpyToSymbol(d_omega, &h_omega, sizeof(float));
                break;
            case 'c':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_force, &ftmp, sizeof(float));
                break;
            case 'd':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_gam, &ftmp, sizeof(float));
                break;
            case 'e':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dg, &ftmp, sizeof(float));
                break;
/*            case 'f':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(float));
                break;
            case 'g':
                h_lambda = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &h_lambda, sizeof(float));
                break;
            case 'h':
                h_comp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &h_comp, sizeof(int));
                break;*/
            case 'i':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'j':
                h_block = atoi(optarg);
                break;
            case 'k':
                h_paths = atoi(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(int));
                break;
            case 'l':
                h_periods = atoi(optarg);
                break;
            case 'm':
                h_trans = atoi(optarg);
                break;
            case 'n':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'o':
                h_samples = atoi(optarg);
                cudaMemcpyToSymbol(d_samples, &h_samples, sizeof(int));
                break;
            case 'p':
                if ( !strcmp(optarg, "predcorr") )
                    h_2ndorder = 1;
                else if ( !strcmp(optarg, "euler") )
                    h_2ndorder = 0;
                cudaMemcpyToSymbol(d_2ndorder, &h_2ndorder, sizeof(int));
                break;
            case 'q':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                    h_traj = 0;
                    h_hist = 0;
                } else if ( !strcmp(optarg, "trajectory") ) {
                    h_moments = 0;
                    h_traj = 1;
                    h_hist = 0;
                } else if ( !strcmp(optarg, "histogram") ) {
                    h_moments = 0;
                    h_traj = 0;
                    h_hist = 1;
                }
                cudaMemcpyToSymbol(d_moments, &h_moments, sizeof(int));
                break;
            case 'r':
                h_domain = optarg;
                break;
            case 's':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 't':
                h_domainy = optarg[0];
                break;
            case 'u':
                h_logx = atoi(optarg);
                break;
            case 'v':
                h_logy = atoi(optarg);
                break;
            case 'w':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'y':
                h_beginx = atof(optarg);
                break;
            case 'z':
                h_endx = atof(optarg);
                break;
            case 'A':
                h_beginy = atof(optarg);
                break;
            case 'B':
                h_endy = atof(optarg);
                break;
/*            case 'C':
                h_mean = atof(optarg);
                cudaMemcpyToSymbol(d_mean, &h_mean, sizeof(float));
                break;
            case 'D':
                h_fa = atof(optarg);
                cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(float));
                break;
            case 'E':
                h_fb = atof(optarg);
                cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(float));
                break;
            case 'F':
                h_mua = atof(optarg);
                cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(float));
                break;
            case 'G':
                h_mub = atof(optarg);
                cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(float));
                break;*/
        }
    }
}

//initialize device random number generator
__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], 0, 0, &d_states[idx]);
}

__device__ float drift(float l_x, float l_v, float l_w, float l_gam, float l_amp, float l_force)
{
    return -l_gam*l_v - 2.0f*PI*cosf(2.0f*PI*l_x) + l_amp*cosf(l_w) + l_force;
}

__device__ float diffusion(float l_gam, float l_Dg, float l_dt, int l_2ndorder, curandState *l_state)
{
    if (l_Dg != 0.0f) {
        float r = curand_uniform(l_state);
        float g = sqrtf(2.0f*l_gam*l_Dg);
        if (l_2ndorder) {
            if ( r <= 1.0f/6 ) {
                return -g*sqrtf(3.0f*l_dt);
            } else if ( r > 1.0f/6 && r <= 2.0f/6 ) {
                return g*sqrtf(3.0f*l_dt);
            } else {
                return 0.0f;
            }
        } else {
            if ( r <= 0.5f ) {
                return -g*sqrtf(l_dt);
            } else {
                return g*sqrtf(l_dt);
            }
        }
    } else {
        return 0.0f;
    }
}

/*__device__ float adapted_jump_poisson(int &npcd, int pcd, float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_lambda != 0.0f) {
        if (pcd <= 0) {
            float ampmean = sqrtf(l_lambda/l_Dp);
           
            npcd = (int) floorf( -logf( curand_uniform(l_state) )/l_lambda/l_dt + 0.5f );

            if (l_comp) {
                float comp = sqrtf(l_Dp*l_lambda)*l_dt;
                
                return -logf( curand_uniform(l_state) )/ampmean - comp;
            } else {
                return -logf( curand_uniform(l_state) )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                float comp = sqrtf(l_Dp*l_lambda)*l_dt;
                
                return -comp;
            } else {
                return 0.0f;
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float adapted_jump_dich(int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt, curandState *l_state)
{
    if (l_mua != 0.0f || l_mub != 0.0f) {
        if (dcd <= 0) {
            if (dst == 0) {
                ndst = 1; 
                ndcd = (int) floorf( -logf( curand_uniform(l_state) )/l_mub/l_dt + 0.5f );
                return l_fb*l_dt;
            } else {
                ndst = 0;
                ndcd = (int) floorf( -logf( curand_uniform(l_state) )/l_mua/l_dt + 0.5f );
                return l_fa*l_dt;
            }
        } else {
            ndcd = dcd - 1;
            if (dst == 0) {
                return l_fa*l_dt;
            } else {
                return l_fb*l_dt;
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float regular_jump_poisson(float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_lambda != 0.0f) {
        float mu, ampmean, comp, s;
        int i;
        unsigned int n;

        mu = l_lambda*l_dt;
        ampmean = sqrtf(l_lambda/l_Dp);
        comp = sqrtf(l_Dp*l_lambda)*l_dt;
        n = curand_poisson(l_state, mu);
        s = 0.0f;
            for (i = 0; i < n; i++) {
                s += -logf( curand_uniform(l_state) )/ampmean;
            }
        if (l_comp) s -= comp;
        return s;
    } else {
        return 0.0f;
    }
}*/

/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
__device__ void predcorr(float &corrl_x, float l_x, float &corrl_v, float l_v, float &corrl_w, float l_w, int &npcd, int pcd, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_gam, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
{
    float l_xt, l_xtt, l_vt, l_vtt, l_wt, l_wtt, predl_x, predl_v, predl_w;

    l_xt = l_v;
    l_vt = drift(l_x, l_v, l_w, l_gam, l_amp, l_force);
    l_wt = l_omega;

    predl_x = l_x + l_xt*l_dt;
    predl_v = l_v + l_vt*l_dt + diffusion(l_gam, l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + l_wt*l_dt;

    l_xtt = predl_v;
    l_vtt = drift(predl_x, predl_v, predl_w, l_gam, l_amp, l_force);
    l_wtt = l_omega;

    predl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt;
    predl_v = l_v + 0.5f*(l_vt + l_vtt)*l_dt + diffusion(l_gam, l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;

    l_xtt = predl_v;
    l_vtt = drift(predl_x, predl_v, predl_w, l_gam, l_amp, l_force);
    l_wtt = l_omega;

    corrl_x = l_x + 0.5f*(l_xt + l_xtt)*l_dt;
    corrl_v = l_v + 0.5f*(l_vt + l_vtt)*l_dt + diffusion(l_gam, l_Dg, l_dt, l_2ndorder, l_state) + adapted_jump_poisson(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt, l_state) + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state);
    corrl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;
}

/* simplified weak order 1.0 regular euler-maruyama scheme 
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 508, 
  C. Kim, E. Lee, P. Talkner, and P.Hanggi; Phys. Rev. E 76; 011109; 2007 ) 
*/
__device__ void eulermaruyama(float &nl_x, float l_x, float &nl_v, float l_v, float &nl_w, float l_w, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_gam, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, \
                         int &ndcd, int dcd, int &ndst, int dst, float l_fa, float l_fb, float l_mua, float l_mub, float l_dt)
{
    float l_xt, l_vt, l_wt;

    l_vt = l_v + drift(l_x, l_v, l_w, l_gam, l_amp, l_force)*l_dt
               + diffusion(l_gam, l_Dg, l_dt, l_2ndorder, l_state)
               + regular_jump_poisson(l_lambda, l_Dp, l_comp, l_dt, l_state)
               + adapted_jump_dich(ndcd, dcd, ndst, dst, l_fa, l_fb, l_mua, l_mub, l_dt, l_state);
    l_xt = l_x + l_v*l_dt;
    l_wt = l_w + l_omega*l_dt;

    nl_v = l_vt;
    nl_x = l_xt;
    nl_w = l_wt;
}

//reduce periodic variable to the base domain
__global__ void fold(float *d_x, float *d_fx, float p)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x, l_fx, f;

    l_x = d_x[idx];
    l_fx = d_fx[idx];

    if (fabsf(l_x) >= p) {
        f = floorf(l_x/p)*p;
        l_x = l_x - f;
        l_fx = l_fx + f;
    }

    d_x[idx] = l_x;
    d_fx[idx] = l_fx;
}

//unfold periodic variable
void unfold(float *x, float *fx)
{
    long i;

    for (i = 0; i < h_threads; i++) {
        x[i] = x[i] + fx[i];
    }
}

//actual simulation kernel
__global__ void run_sim(float *d_x1, float *d_x2, float *d_v1, float *d_v2; float *d_w, float *d_sv1, float *d_sv2, float *d_sv12, float *d_sv22, float *d_dx, curandState *d_states); //int *d_pcd, int *d_dcd, int *d_dst, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    //cache path and model parameters in local variables
    float l_x1, l_x2, l_v1, l_v2, l_w, l_sv1, l_sv2, l_sv12, l_sv22, l_dx; 
    curandState l_state;

    l_x1 = d_x1[idx];
    l_x2 = d_x2[idx];
    l_v1 = d_v1[idx];
    l_v2 = d_v2[idx];
    l_w = d_w[idx];
    l_sv1 = d_sv1[idx];
    l_sv2 = d_sv2[idx];
    l_sv12 = d_sv12[idx];
    l_sv22 = d_sv22[idx];
    l_state = d_states[idx];

    float l_amp, l_omega, l_force, l_gam, l_Dg;//, l_Dp, l_lambda, l_mean, l_fa, l_fb, l_mua, l_mub;
    int l_2ndorder;//, l_comp;
    int l_moments;

    l_amp = d_amp;
    l_omega = d_omega;
    l_force = d_force;
    l_gam = d_gam;
    l_Dg = d_Dg;
    l_Dp = d_Dp;
/*    l_lambda = d_lambda;
    l_mean = d_mean;
    l_comp = d_comp;
    l_fa = d_fa;
    l_fb = d_fb;
    l_mua = d_mua;
    l_mub = d_mub;*/
    l_2ndorder = d_2ndorder;
    l_moments = d_moments;
   
    //run simulation for multiple values of the system parameters
    if (l_moments) {
        long ridx = (idx/d_paths) % d_points;
        l_dx = d_dx[ridx];

        switch(d_domainx) {
            case 'a':
                l_amp = l_dx;
                break;
            case 'w':
                l_omega = l_dx;
                break;
            case 'f':
                l_force = l_dx;
                break;
            case 'g':
                l_gam = l_dx;
                break;
            case 'D':
                l_Dg = l_dx;
                break;
/*            case 'p':
                l_Dp = l_dx;
                if (l_mean != 0.0f) l_lambda = (l_mean*l_mean)/l_Dp;
                break;
            case 'l':
                l_lambda = l_dx;
                if (l_mean != 0.0f) l_Dp = (l_mean*l_mean)/l_lambda;
                break;
            case 'i':
                l_fa = l_dx;
                if (l_comp == 1) {
                    l_mua = -l_fa*l_mub/l_fb;
                    //l_fb = -l_fa*l_mub/l_mua;
                } else if (l_mean != 0.0f) {
                    l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
                    //l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
                }
                break;
            case 'j':
                l_fb = l_dx;
                if (l_comp == 1) {
                    l_mub = -l_fb*l_mua/l_fa;
                    //l_fa = -l_fb*l_mua/l_mub;
                } else if (l_mean != 0.0f) {
                    l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
                    //l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
                }
                break;
            case 'm':
                l_mua = l_dx;
                if (l_comp == 1) {
                    l_fa = -l_fb*l_mua/l_mub;
                    //l_mub = -l_fb*l_mua/l_fa;
                } else if (l_mean != 0.0f) {
                    l_fa = (l_mean*(l_mua + l_mub) - l_fb*l_mua)/l_mub;
                    //l_mub = (l_fb - l_mean)*l_mua/(l_mean - l_fa);
                }
                break;
            case 'n':
                l_mub = l_dx;
                if (l_comp == 1) {
                    l_fb = -l_fa*l_mub/l_mua;
                    //l_mua = -l_fa*l_mub/l_fb;
                } else if (l_mean != 0.0f) {
                    l_fb = (l_mean*(l_mua + l_mub) - l_fa*l_mub)/l_mua;
                    //l_mua = (l_fa - l_mean)*l_mub/(l_mean - l_fb);
                }
                break;*/
        }
    }

    //step size
    float l_dt, tmp;

    l_dt = 2.0f*PI/l_omega;

/*    if (l_lambda != 0.0f) {
        if (l_2ndorder) {
            tmp = 1.0f/l_lambda;
            if (tmp < l_dt) l_dt = tmp;
        }
    }*

    if (l_mua != 0.0f || l_mub != 0.0f) {
        float taua, taub;

        taua = 1.0f/l_mua;
        taub = 1.0f/l_mub;
        
        if (taua < taub) {
            tmp = taua;
        } else {
            tmp = taub;
        }

        if (tmp < l_dt) l_dt = tmp;
    }*/

    int l_spp;

    l_spp = d_spp;
    l_dt /= l_spp;

    //number of steps
    int l_samples;

    l_samples = d_samples;

/*    //jump countdowns
    int l_initnoise, l_pcd, l_dcd, l_dst;

    l_initnoise = d_initnoise;

    if (l_initnoise) {

        if (l_lambda != 0.0f) {
            if (l_2ndorder) {
                l_pcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );
            }
        }

        if (l_mua != 0.0f || l_mub != 0.0f) {
            float rn;
            rn = curand_uniform(&l_state);

            if (rn < 0.5f) {
                l_dst = 0;
                l_dcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_mua/l_dt + 0.5f);
            } else {
                l_dst = 1;
                l_dcd = (int) floorf( -logf( curand_uniform(&l_state) )/l_mub/l_dt + 0.5f);
            }
        }

    } else {
        
        if (l_lambda != 0.0f) {
            if (l_2ndorder) {
                l_pcd = d_pcd[idx];
            }
        }
    
        if (l_mua != 0.0f || l_mub != 0.0f) {
            l_dcd = d_dcd[idx];
            l_dst = d_dst[idx];
        }*/

        int i, l_trigger;

        l_trigger = d_trigger;
    
        for (i = 0; i < l_samples; i++) {
            //algorithm
            if (l_2ndorder) {
                predcorr(l_x, l_x, l_v, l_v, l_w, l_w, l_pcd, l_pcd, &l_state, l_amp, l_omega, l_force, l_gam, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                         l_dcd, l_dcd, l_dst, l_dst, l_fa, l_fb, l_mua, l_mub, l_dt);
            } else {
                eulermaruyama(l_x, l_x, l_v, l_v, l_w, l_w, &l_state, l_amp, l_omega, l_force, l_gam, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, \
                         l_dcd, l_dcd, l_dst, l_dst, l_fa, l_fb, l_mua, l_mub, l_dt);
            }
        
            if (l_moments) {
                if (l_trigger) {
                    l_sv1 += l_v1;
                    l_sv2 += l_v2;
                    l_sv12 += l_v1*l_v1;
                    l_sv22 += l_v2*l_v2;
                }
            }
        }
//    }

    //write back path parameters to the global memory
    d_x1[idx] = l_x1;
    d_x2[idx] = l_x2;
    d_v1[idx] = l_v1;
    d_v2[idx] = l_v2;
    d_w[idx] = l_w;
    d_sv1[idx] = l_sv1;
    d_sv2[idx] = l_sv2;
    d_sv12[idx] = l_sv12;
    d_sv22[idx] = l_sv22;
/*    d_pcd[idx] = l_pcd;
    d_dcd[idx] = l_dcd;
    d_dst[idx] = l_dst;*/
    d_states[idx] = l_state;
}

//prepare simulation
void prepare()
{
    //grid size
    h_paths = (h_paths/h_block)*h_block;
    h_threads = h_paths;

    if (h_moments) h_threads *= h_points;

    h_grid = h_threads/h_block;

    //number of steps
    h_steps = h_periods*h_spp;
     
    //host memory allocation
    size_f = h_threads*sizeof(float);
    size_i = h_threads*sizeof(int);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(float);

    h_x1 = (float*)malloc(size_f);
    h_x2 = (float*)malloc(size_f);
    h_fx1 = (float*)malloc(size_f);
    h_fx2 = (float*)malloc(size_f);
    h_v1 = (float*)malloc(size_f);
    h_v2 = (float*)malloc(size_f);
    h_w = (float*)malloc(size_f);
    h_fw = (float*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x1, size_f);
    cudaMalloc((void**)&d_x2, size_f);
    cudaMalloc((void**)&d_fx1, size_f);
    cudaMalloc((void**)&d_fx2, size_f);
    cudaMalloc((void**)&d_v1, size_f);
    cudaMalloc((void**)&d_v2, size_f);
    cudaMalloc((void**)&d_w, size_f);
    cudaMalloc((void**)&d_fw, size_f);
/*    cudaMalloc((void**)&d_pcd, size_i);
    cudaMalloc((void**)&d_dcd, size_i);
    cudaMalloc((void**)&d_dst, size_i);*/
    cudaMalloc((void**)&d_seeds, size_ui);
    cudaMalloc((void**)&d_states, h_threads*sizeof(curandState));

    //copy seeds from host to device
    cudaMemcpy(d_seeds, h_seeds, size_ui, cudaMemcpyHostToDevice);

    //initialization of device rng
    init_dev_rng<<<h_grid, h_block>>>(d_seeds, d_states);

    free(h_seeds);
    cudaFree(d_seeds);

    //moments specific requirements
    h_sv1 = (float*)malloc(size_f);
    h_sv2 = (float*)malloc(size_f);
    h_sv12 = (float*)malloc(size_f);
    h_sv22 = (float*)malloc(size_f);
    h_dx = (float*)malloc(size_p);

    float dxtmp = h_beginx;
    float dxstep = (h_endx - h_beginx)/h_points;

    int i;
        
    //set domainx
    for (i = 0; i < h_points; i++) {
        if (h_logx) {
            h_dx[i] = exp10f(dxtmp);
        } else {
            h_dx[i] = dxtmp;
        }
        dxtmp += dxstep;
    }
        
    cudaMalloc((void**)&d_sv1, size_f);
    cudaMalloc((void**)&d_sv2, size_f);
    cudaMalloc((void**)&d_sv12, size_f);
    cudaMalloc((void**)&d_sv22, size_f);
    cudaMalloc((void**)&d_dx, size_p);
    
    cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
}

void copy_to_dev()
{
    cudaMemcpy(d_x1, h_x1, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx1, h_fx1, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx2, h_fx2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, h_v1, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, h_v2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fw, h_fw, size_f, cudaMemcpyHostToDevice);

    cudaMemcpy(d_sv1, h_sv1, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv2, h_sv2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv12, h_sv12, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv22, h_sv22, size_f, cudaMemcpyHostToDevice);
}

void copy_from_dev()
{
    cudaMemcpy(h_x1, d_x1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx1, d_fx1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx2, d_fx2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v1, d_v1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v2, d_v2, size_f, cudaMemcpyDeviceToHost);
}

//set initial conditions
void initial_conditions()
{
    curandGenerateUniform(gen, h_x1, h_threads); //x1 in (0,1]
    curandGenerateUniform(gen, h_x2, h_threads); //x2 in (0,1]
    curandGenerateUniform(gen, h_v1, h_threads);
    curandGenerateUniform(gen, h_v2, h_threads);
    curandGenerateUniform(gen, h_w, h_threads);

    long i;

    for (i = 0; i < h_threads; i++) {
        h_v1[i] = 4.0f*h_v1[i] - 2.0f; //v1 in (-2,2]
        h_v2[i] = 4.0f*h_v2[i] - 2.0f; //v2 in (-2,2]
        h_w[i] *= 2.0f*PI; //w in (0,2\pi]
    }

    memset(h_fx1, 0.0f, size_f);
    memset(h_fx2, 0.0f, size_f);
    memset(h_fw, 0.0f, size_f);
    
    memset(h_sv1, 0.0f, size_f);
    memset(h_sv2, 0.0f, size_f);
    memset(h_sv12, 0.0f, size_f);
    memset(h_sv22, 0.0f, size_f);

    copy_to_dev();
}

//calculate the first two moments of <v> and diffusion coefficient
void moments(float *av1, float *av2, float *av12, float *av22, float *dc1, float *dc2)
{
    float sv1, sv2, sv12, sv22, sx1, sx2, sx12, sx22, dt, tempo;//, tmp, taua, taub;
    int i, j;

    cudaMemcpy(h_sv1, d_sv1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sv2, d_sv2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sv12, d_sv12, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sv22, d_sv22, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x1, d_x1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx1, d_fx1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fx2, d_fx2, size_f, cudaMemcpyDeviceToHost);

    unfold(h_x1, h_fx1);
    unfold(h_x2, h_fx2);

    for (j = 0; j < h_points; j++) {
        sv1 = 0.0f;
        sv2 = 0.0f;
        sv12 = 0.0f;
        sv22 = 0.0f;
        sx1 = 0.0f;
        sx2 = 0.0f;
        sx12 = 0.0f;
        sx22 = 0.0f;

        for (i = 0; i < h_paths; i++) {
            sv1 += h_sv1[j*h_paths + i];
            sv2 += h_sv2[j*h_paths + i];
            sv12 += h_sv12[j*h_paths + i];
            sv22 += h_sv22[j*h_paths + i];
            sx1 += h_x1[j*h_paths + i];
            sx2 += h_x2[j*h_paths + i];
            sx12 += h_x1[j*h_paths + i]*h_x1[j*h_paths + i];
            sx22 += h_x2[j*h_paths + i]*h_x2[j*h_paths + i];
        }

        av1[j] = sv1/((h_periods - h_trans)*h_spp)/h_paths;
        av2[j] = sv2/((h_periods - h_trans)*h_spp)/h_paths;
        av12[j] = sv12/((h_periods - h_trans)*h_spp)/h_paths;
        av22[j] = sv22/((h_periods - h_trans)*h_spp)/h_paths;

        //external driving
        if (h_domainx == 'w') {
            tempo = 2.0f*PI/h_dx[j];
        } else {
            tempo = 2.0f*PI/h_omega;
        }
       
        dt = tempo;

/*        //Poissonian
        if (h_lambda != 0.0f && h_2ndorder) {
            if (h_domainx == 'l') {
                tmp = 1.0f/h_dx[j];
            } else if (h_domainx == 'p' && h_mean != 0.0f) {
                tmp = 1.0f/(h_mean*h_mean/h_dx[j]);
            } else {
                tmp = 1.0f/h_lambda;
            }

            if (tmp < tempo) dt = tmp;
        }

        //Dichotomous
        if (h_mua != 0.0f || h_mub != 0.0f) {
            if (h_domainx == 'm') {
                taua = 1.0f/h_dx[j];
                taub = 1.0f/h_mub;

                if (h_comp) {
                    tmp = 1.0f/(-h_fb*h_dx[j]/h_fa);
                } else if (h_mean != 0.0f) {
                    tmp = 1.0f/((h_fb - h_mean)*h_dx[j]/(h_mean - h_fa));
                } else {
                    tmp = taub;
                //}
            
                if (taua <= tmp) {
                    if (taua < tempo) dt = taua;
                } else {
                    if (tmp < tempo) dt = tmp;
                }
            } else if (h_domainx == 'n') {
                taua = 1.0f/h_mua;
                taub = 1.0f/h_dx[j];

                if (h_comp) {
                    tmp = 1.0f/(-h_fa*h_dx[j]/h_fb);
                } else if (h_mean != 0.0f) {
                    tmp = 1.0f/((h_fa - h_mean)*h_dx[j]/(h_mean - h_fb));
                } else {
                    tmp = taua;
                //}

                if (taub <= tmp) {
                    if (taub < tempo) dt = taub;
                } else {
                    if (tmp < tempo) dt = tmp;
                }
            } else if (h_domainx == 'i') {
                taua = 1.0f/h_mua;
                taub = 1.0f/h_mub;

                if (h_comp) {
                    tmp = 1.0f/(-h_dx[j]*h_mub/h_fb);
                } else if (h_mean != 0.0f) {
                    tmp = 1.0f/((h_dx[j] - h_mean)*h_mub/(h_mean - h_fb));
                } else {
                    tmp = taua;
                }

                if (taub <= tmp) {
                    if (taub < tempo) dt = taub;
                } else {
                    if (tmp < tempo) dt = tmp;
                }
            } else if (h_domainx == 'j') {
                taua = 1.0f/h_mua;
                taub = 1.0f/h_mub;

                if (h_comp) {
                    tmp = 1.0f/(-h_dx[j]*h_mua/h_fa);
                } else if (h_mean != 0.0f) {
                    tmp = 1.0f/((h_dx[j] - h_mean)*h_mua/(h_mean - h_fa));
                } else {
                    tmp = taub;
                }

                if (taua <= tmp) {
                    if (taua < tempo) dt = taua;
                } else {
                    if (tmp < tempo) dt = tmp;
                }
            } else {
                taua = 1.0f/h_mua;
                taub = 1.0f/h_mub;

                if (taua < taub) {
                    if (taua < tempo) dt = taua;
                } else {
                    if (taub < tempo) dt = taub;
                }
            }
        }*/

        dt /= h_spp;

        sx1 /= h_paths;
        sx2 /= h_paths;
        sx12 /= h_paths;
        sx22 /= h_paths;
        dc1[j] = (sx12 - sx1*sx1)/(2.0f*h_steps*dt);
        dc2[j] = (sx22 - sx2*sx2)/(2.0f*h_steps*dt);
    }
}

//calculate ensemble average
void ensemble_average(float *h_x1, *float h_x2, float *h_v1, float *h_v2, float &sx1, float &sx2, float &sx12, float &sx22, float &sv1, float &sv2, float &sv12, float &sv22)
{
    long i;

    sx1 = 0.0f;
    sx2 = 0.0f;
    sv1 = 0.0f;
    sv2 = 0.0f;
    sx12 = 0.0f;
    sx22 = 0.0f;
    sv12 = 0.0f;
    sv22 = 0.0f;

    for (i = 0; i < h_threads; i++) {
        sx1 += h_x1[i];
        sx2 += h_x2[i];
        sv1 += h_v1[i];
        sv2 += h_v2[i];
        sx12 += h_x1[i]*h_x1[i];
        sx22 += h_x2[i]*h_x2[i];
        sv12 += h_v1[i]*h_v1[i];
        sv22 += h_v2[i]*h_v2[i];
    }

    sx1 /= h_threads;
    sx2 /= h_threads;
    sv1 /= h_threads;
    sv2 /= h_threads;
    sx12 /= h_threads;
    sx22 /= h_threads;
    sv12 /= h_threads;
    sv22 /= h_threads;
}

//free memory
void finish()
{
    free(h_x1);
    free(h_x2);
    free(h_fx1);
    free(h_fx2);
    free(h_v1);
    free(h_v2);
    free(h_w);
    free(h_fw);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_fx1);
    cudaFree(d_fx2);
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_w);
    cudaFree(d_fw);
/*    cudaFree(d_pcd);
    cudaFree(d_dcd);
    cudaFree(d_dst);*/
    cudaFree(d_states);
    
    free(h_sv1);
    free(h_sv2);
    free(h_sv12);
    free(h_sv22);
    free(h_dx);

    cudaFree(d_sv1);
    cudaFree(d_sv2);
    cudaFree(d_sv12);
    cudaFree(d_sv22);
    cudaFree(d_dx);
}

int main(int argc, char **argv)
{
    parse_cla(argc, argv);
    if (!h_moments && !h_traj && !h_hist) {
        usage(argv);
        return -1;
    }

    prepare();

    initial_conditions();

/*    h_initnoise = 0;
    cudaMemcpyToSymbol(d_initnoise, &h_initnoise, sizeof(int));

    if ( (h_lambda != 0.0f && h_2ndorder) || (h_mua != 0.0f || h_mub != 0.0f) ) {
        h_initnoise = 1;
        cudaMemcpyToSymbol(d_initnoise, &h_initnoise, sizeof(int));

        run_sim<<<h_grid, h_block>>>(d_x, d_v, d_w, d_sv, d_sv2, d_dx, d_pcd, d_dcd, d_dst, d_states);

        h_initnoise = 0;
        cudaMemcpyToSymbol(d_initnoise, &h_initnoise, sizeof(int));
    }*/

    //asymptotic long time average velocity <<v>>, <<v^2>> and diffusion coefficient
    if (h_moments) {
        float *av1, *av2, *av12, *av22, *dc1, *dc2;
        long i;

        av1 = (float*)malloc(size_p);
        av2 = (float*)malloc(size_p);
        av12 = (float*)malloc(size_p);
        av22 = (float*)malloc(size_p);
        dc1 = (float*)malloc(size_p);
        dc2 = (float*)malloc(size_p);

        if ( !strcmp(h_domain, "1d") ) { 

            h_trigger = 0;
            cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(int));

            for (i = 0; i < h_steps; i += h_samples) {
                run_sim<<<h_grid, h_block>>>(d_x1, d_x2, d_v1, d_v2, d_w, d_sv1, d_sv2, d_sv12, d_sv22, d_dx, d_states);//d_pcd, d_dcd, d_dst, d_states);
                fold<<<h_grid, h_block>>>(d_x1, d_fx1, 1.0f);
                fold<<<h_grid, h_block>>>(d_x2, d_fx2, 1.0f);
                fold<<<h_grid, h_block>>>(d_w, d_fw, (2.0f*PI));
                if (i == h_trans*h_spp) {
                    h_trigger = 1;
                    cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(int));
                }
            }

            moments(av1, av2, av12, av22, dc1, dc2);

            printf("#%c <<v1>> <<v1^2>> <<v2>> <<v2^2>> D_x1 D_x2\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e %e %e %e %e %e\n", h_dx[i], av1[i], av12[i], av2[i], av22[i], dc1[i], dc2[i]);
            }

        } else {
            float h_dy, dytmp, dystep;
            int j, k;
            
            dytmp = h_beginy;
            dystep = (h_endy - h_beginy)/h_points;
            
            printf("#%c %c <<v1>> <<v1^2>> <<v2>> <<v2^2>> D_x1 D_x2\n", h_domainx, h_domainy);
            
            for (k = 0; k < h_points; k++) {
                if (h_logy) {
                    h_dy = exp10f(dytmp);
                } else {
                    h_dy = dytmp;
                }

                switch(h_domainy) {
                    case 'a':
                        cudaMemcpyToSymbol(d_amp, &h_dy, sizeof(float));
                        break;
                    case 'w':
                        h_omega = h_dy;
                        cudaMemcpyToSymbol(d_omega, &h_omega, sizeof(float));
                        break;
                    case 'f':
                        cudaMemcpyToSymbol(d_force, &h_dy, sizeof(float));
                        break;
                    case 'g':
                        cudaMemcpyToSymbol(d_gam, &h_dy, sizeof(float));
                        break;
                    case 'D':
                        cudaMemcpyToSymbol(d_Dg, &h_dy, sizeof(float));
                        break;
/*                    case 'p':
                        cudaMemcpyToSymbol(d_Dp, &h_dy, sizeof(float));
                        break;
                    case 'l':
                        h_lambda = h_dy;
                        cudaMemcpyToSymbol(d_lambda, &h_lambda, sizeof(float));
                        break;
                    case 'i':
                        h_fa = h_dy;
                        cudaMemcpyToSymbol(d_fa, &h_fa, sizeof(float));
                        break;
                    case 'j':
                        h_fb = h_dy;
                        cudaMemcpyToSymbol(d_fb, &h_fb, sizeof(float));
                        break;
                    case 'm':
                        h_mua = h_dy;
                        cudaMemcpyToSymbol(d_mua, &h_mua, sizeof(float));
                        break;
                    case 'n':
                        h_mub = h_dy;
                        cudaMemcpyToSymbol(d_mub, &h_mub, sizeof(float));
                        break;*/
                }
 
                h_trigger = 0;
                cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(int));

                for (i = 0; i < h_steps; i += h_samples) {
                    run_sim<<<h_grid, h_block>>>(d_x1, d_x2, d_v1, d_v2 d_w, d_sv1, d_sv2, d_sv12, d_sv22, d_dx, d_states);//d_pcd, d_dcd, d_dst, d_states);
                    fold<<<h_grid, h_block>>>(d_x1, d_fx1, 1.0f);
                    fold<<<h_grid, h_block>>>(d_x2, d_fx2, 1.0f);
                    fold<<<h_grid, h_block>>>(d_w, d_fw, (2.0f*PI));
                    if (i == h_trans*h_spp) {
                        h_trigger = 1;
                        cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(int));
                    }
                }
 
                moments(av1, av2, av12, av22, dc1, dc2);
                
                for (j = 0; j < h_points; j++) {
                    printf("%e %e %e %e %e %e %e %e\n", h_dx[j], h_dy, av1[j], av12[j], av2[j], av22[j], dc1[j], dc2[j]);
                }

                //blank line for plotting purposes
                printf("\n");

                initial_conditions();

/*                if ( (h_lambda != 0.0f && h_2ndorder) || (h_mua != 0.0f || h_mub != 0.0f) ) {
                    h_initnoise = 1;
                    cudaMemcpyToSymbol(d_initnoise, &h_initnoise, sizeof(int));

                    run_sim<<<h_grid, h_block>>>(d_x, d_v, d_w, d_sv, d_sv2, d_dx, d_pcd, d_dcd, d_dst, d_states);

                    h_initnoise = 0;
                    cudaMemcpyToSymbol(d_initnoise, &h_initnoise, sizeof(int));
                }*/

                dytmp += dystep;
           }
        }

        free(av1);
        free(av2);
        free(av12);
        free(av22);
        free(dc1);
        free(dc2);
    }

    //ensemble averaged trajectory <x>(t), <v>(t) and <x^2>(t), <v^2>(t)
    if (h_traj) {
        float t, sx1, sx2, sv1, sv2, sx12, sx22, sv12, sv22, dt, tmp;//, taua, taub;
        long i;

        dt = 2.0f*PI/h_omega;
        tmp = dt;

/*        if (h_lambda != 0.0f && h_2ndorder) tmp = 1.0f/h_lambda;

        if (h_mua != 0.0f || h_mub != 0.0f) {
            taua = 1.0f/h_mua;
            taub = 1.0f/h_mub;

            if (taua < taub) {
                tmp = taua;
            } else {
                tmp = taub;
            }
        }

        if (tmp < dt) dt = tmp;*/

        dt /= h_spp;

        printf("#t <x1> <v1> <x1^2> <v1^2> <x2> <v2> <x2^2> <v2^2>\n");
        
        for (i = 0; i < h_steps; i += h_samples) {
            run_sim<<<h_grid, h_block>>>(d_x1, d_x2, d_v1, d_v2, d_w, d_sv1, d_sv2, d_sv12, d_sv22, d_dx, d_states);//d_pcd, d_dcd, d_dst, d_states);
            copy_from_dev();
            unfold(h_x1, h_fx1);
            unfold(h_x2, h_fx2);
            t = i*dt;
            ensemble_average(h_x1, h_x2, h_v1, h_v2, sx1, sx2, sx12, sx22, sv1, sv2, sv12, sv22);
            printf("%e %e %e %e %e %e %e %e %e\n", t, sx1, sv1, sx12, sv12, sx2, sv2, sx22, sv22);
            fold<<<h_grid, h_block>>>(d_x1, d_fx1, 1.0f);
            fold<<<h_grid, h_block>>>(d_x2, d_fx2, 1.0f);
            fold<<<h_grid, h_block>>>(d_w, d_fw, (2.0f*PI));
        }
    }

    //the final position x and velocity v of all paths
    if (h_hist) {
        long i;

        for (i = 0; i < h_steps; i += h_samples) {
            run_sim<<<h_grid, h_block>>>(d_x1, d_x2, d_v1, d_v2, d_w, d_sv1, d_sv2, d_sv12, d_sv22, d_dx, d_states);//d_pcd, d_dcd, d_dst, d_states);
            fold<<<h_grid, h_block>>>(d_x1, d_fx1, 1.0f);
            fold<<<h_grid, h_block>>>(d_x2, d_fx2, 1.0f);
            fold<<<h_grid, h_block>>>(d_w, d_fw, (2.0f*PI));
        }
        
        copy_from_dev();

        printf("#x1 v1 x2 v2\n");
        
        for (i = 0; i < h_threads; i++) {
            printf("%e %e %e %e\n", h_x1[i], h_v1[i], h_x2[i], h_v2[i]); 
        }
    }

    finish();

    return 0;
}
