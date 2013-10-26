/*
 * Two identical overdamped AC-driven Josephson junctions coupled by external resistive shunt
 *
 * \dot{\phi}_1 = - sin(\phi_1) + i_1(t) + \alpha[I_2(t) - sin(\phi_2)] + \sqrt{Dg}\eta_1(t)
 * \dot{\phi}_2 = - sin(\phi_2) + i_2(t) + \alpha[I_1(t) - sin(\phi_1)] + \sqrt{Dg}\eta_2(t)
 *
 * see J. Spiechowicz et al., Acta Phys. Polon. B 43, 1203 (2012)
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
__constant__ float d_amp, d_omega, d_force, d_alpha, d_Dg, d_Dp, d_lambda;
__constant__ int d_comp;
float h_omega;

//simulation
float h_trans;
int h_dev, h_block, h_grid, h_spp;
long h_paths, h_periods, h_threads, h_steps, h_trigger;
__constant__ int d_spp, d_2ndorder;
__constant__ long d_paths, d_steps, d_trigger;

//output
char *h_domain;
char h_domainx, h_domainy;
float h_beginx, h_endx, h_beginy, h_endy;
int h_logx, h_logy, h_points, h_moments, h_traj, h_hist;
__constant__ char d_domainx;
__constant__ int d_points;

//vector
float *h_x1, *h_x2, *h_w, *h_x1b, *h_x2b, *h_dx;
float *d_x1, *d_x2, *d_w, *d_x1b, *d_x2b, *d_dx;
unsigned int *h_seeds, *d_seeds;
curandState *d_states;

size_t size_f, size_ui, size_p;
curandGenerator_t gen;

static struct option options[] = {
    {"amp", required_argument, NULL, 'a'},
    {"omega", required_argument, NULL, 'b'},
    {"force", required_argument, NULL, 'c'},
    {"alpha", required_argument, NULL, 'd'},
    {"Dg", required_argument, NULL, 'e'},
    {"Dp", required_argument, NULL, 'f'},
    {"lambda", required_argument, NULL, 'g'},
    {"comp", required_argument, NULL, 'h'},
    {"dev", required_argument, NULL, 'i'},
    {"block", required_argument, NULL, 'j'},
    {"paths", required_argument, NULL, 'k'},
    {"periods", required_argument, NULL, 'l'},
    {"trans", required_argument, NULL, 'm'},
    {"spp", required_argument, NULL, 'n'},
    {"algorithm", required_argument, NULL, 'o'},
    {"mode", required_argument, NULL, 'p'},
    {"domain", required_argument, NULL, 'q'},
    {"domainx", required_argument, NULL, 'r'},
    {"domainy", required_argument, NULL, 's'},
    {"logx", required_argument, NULL, 't'},
    {"logy", required_argument, NULL, 'u'},
    {"points", required_argument, NULL, 'v'},
    {"beginx", required_argument, NULL, 'w'},
    {"endx", required_argument, NULL, 'y'},
    {"beginy", required_argument, NULL, 'z'},
    {"endy", required_argument, NULL, 'A'}
};

void usage(char **argv)
{
    printf("Usage: %s <params> \n\n", argv[0]);
    printf("Model params:\n");
    printf("    -a, --amp=FLOAT         set the AC driving amplitude 'amp' to FLOAT\n");
    printf("    -b, --omega=FLOAT       set the AC driving frequency '\\omega' to FLOAT\n");
    printf("    -c, --force=FLOAT       set the external bias 'force' to FLOAT\n");
    printf("    -d, --alpha=FLOAT       set the coupling '\\alpha' to FLOAT\n");
    printf("    -e, --Dg=FLOAT          set the Gaussian noise intensity 'Dg' to FLOAT\n");
    printf("    -f, --Dp=FLOAT          set the Poissonian noise intensity 'Dp' to FLOAT\n");
    printf("    -g, --lambda=FLOAT      set the Poissonian kicks frequency '\\lambda' to FLOAT\n\n");
    printf("    -h, --comp=INT          choose between biased and unbiased Poissonian noise. INT can be one of:\n");
    printf("                            0: biased; 1: unbiased\n");
    printf("Simulation params:\n");
    printf("    -i, --dev=INT           set the gpu device to INT\n");
    printf("    -j, --block=INT         set the gpu block size to INT\n");
    printf("    -k, --paths=LONG        set the number of paths to LONG\n");
    printf("    -l, --periods=LONG      set the number of periods to LONG\n");
    printf("    -m, --trans=FLOAT       specify fraction FLOAT of periods which stands for transients\n");
    printf("    -n, --spp=INT           specify how many integration steps should be calculated\n");
    printf("                            for a single period of the driving force\n\n");
    printf("    -o, --algorithm=STRING  sets the algorithm. STRING can be one of:\n");
    printf("                            predcorr: simplified weak order 2.0 adapted predictor-corrector\n");
    printf("                            euler: simplified weak order 1.0 regular euler-maruyama\n");
    printf("Output params:\n");
    printf("    -p, --mode=STRING       sets the output mode. STRING can be one of:\n");
    printf("                            moments: the first moment <<v>>\n");
    printf("                            trajectory: ensemble averaged <x>(t)\n");
    printf("                            histogram: the final position of all paths\n");
    printf("    -q, --domain=STRING     simultaneously scan over one or two model params. STRING can be one of:\n");
    printf("                            1d: only one parameter; 2d: two parameters at once\n");
    printf("    -r, --domainx=CHAR      sets the first domain of the moments. CHAR can be one of:\n");
    printf("                            a: amp; w: omega, f: force; g: alpha; D: Dg; p: Dp; l: lambda\n");
    printf("    -s, --domainy=CHAR      sets the second domain of the moments (only if --domain=2d). CHAR can be the same as above.\n");
    printf("    -t, --logx=INT          choose between linear and logarithmic scale of the domainx\n");
    printf("                            0: linear; 1: logarithmic\n");
    printf("    -u, --logy=INT          the same as above but for domainy\n");
    printf("    -v, --points=INT        set the number of samples to generate between begin and end\n");
    printf("    -w, --beginx=FLOAT      set the starting value of the domainx to FLOAT\n");
    printf("    -y, --endx=FLOAT        set the end value of the domainx to FLOAT\n");
    printf("    -z, --beginy=FLOAT      the same as --beginx, but for domainy\n");
    printf("    -A, --endy=FLOAT        the same as --endx, but for domainy\n");
    printf("\n");
}

void parse_cla(int argc, char **argv)
{
    float ftmp;
    int c, itmp;

    while( (c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:y:z:A", options, NULL)) != EOF) {
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
                cudaMemcpyToSymbol(d_alpha, &ftmp, sizeof(float));
                break;
            case 'e':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dg, &ftmp, sizeof(float));
                break;
            case 'f':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_Dp, &ftmp, sizeof(float));
                break;
            case 'g':
                ftmp = atof(optarg);
                cudaMemcpyToSymbol(d_lambda, &ftmp, sizeof(float));
                break;
            case 'h':
                itmp = atoi(optarg);
                cudaMemcpyToSymbol(d_comp, &itmp, sizeof(int));
                break;
            case 'i':
                itmp = atoi(optarg);
                cudaSetDevice(itmp);
                break;
            case 'j':
                h_block = atoi(optarg);
                break;
            case 'k':
                h_paths = atol(optarg);
                cudaMemcpyToSymbol(d_paths, &h_paths, sizeof(long));
                break;
            case 'l':
                h_periods = atol(optarg);
                break;
            case 'm':
                h_trans = atof(optarg);
                break;
            case 'n':
                h_spp = atoi(optarg);
                cudaMemcpyToSymbol(d_spp, &h_spp, sizeof(int));
                break;
            case 'o':
                if ( !strcmp(optarg, "predcorr") )
                    itmp = 1;
                else if ( !strcmp(optarg, "euler") )
                    itmp = 0;
                cudaMemcpyToSymbol(d_2ndorder, &itmp, sizeof(int));
                break;
            case 'p':
                if ( !strcmp(optarg, "moments") ) {
                    h_moments = 1;
                    h_traj = 0;
                    h_hist = 0;
                } else if ( !strcmp(optarg, "trajectory") ) {
                    h_traj = 1;
                    h_hist = 0;
                    h_moments = 0;
                } else if ( !strcmp(optarg, "histogram") ) {
                    h_moments = 0;
                    h_traj = 0;
                    h_hist = 1;
                }
                break;
            case 'q':
                h_domain = optarg;
                break;
            case 'r':
                h_domainx = optarg[0]; 
                cudaMemcpyToSymbol(d_domainx, &h_domainx, sizeof(char));
                break;
            case 's':
                h_domainy = optarg[0];
                break;
            case 't':
                h_logx = atoi(optarg);
                break;
            case 'u':
                h_logy = atoi(optarg);
                break;
            case 'v':
                h_points = atoi(optarg);
                cudaMemcpyToSymbol(d_points, &h_points, sizeof(int));
                break;
            case 'w':
                h_beginx = atof(optarg);
                break;
            case 'y':
                h_endx = atof(optarg);
                break;
            case 'z':
                h_beginy = atof(optarg);
                break;
            case 'A':
                h_endy = atof(optarg);
                break;
        }
    }
}

__global__ void init_dev_rng(unsigned int *d_seeds, curandState *d_states)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(d_seeds[idx], idx, 0, &d_states[idx]);
}

__device__ float drift(float l_x, float l_w, float l_amp, float l_force)
{
    return -sinf(l_x) + l_amp*cosf(l_w) + l_force;
}

__device__ float diffusion(float l_Dg, float l_dt, int l_2ndorder, curandState *l_state)
{
    if (l_Dg != 0.0f) {
        float r = curand_uniform(l_state);
        if (l_2ndorder) {
            if ( r <= 1.0f/6 ) {
                return -sqrtf(3.0f*l_Dg*l_dt);
            } else if ( r > 1.0f/6 && r <= 2.0f/6 ) {
                return sqrtf(3.0f*l_Dg*l_dt);
            } else {
                return 0.0f;
            }
        } else {
            if ( r <= 0.5f ) {
                return -sqrtf(l_Dg*l_dt);
            } else {
                return sqrtf(l_Dg*l_dt);
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float adapted_jump(int &npcd, int pcd, float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_Dp != 0.0f) {
        float comp = sqrtf(l_Dp*l_lambda)*l_dt;
        if (pcd <= 0) {
            float ampmean = sqrtf(l_lambda/l_Dp);
           
            npcd = (int) floor( -logf( curand_uniform(l_state) )/l_lambda/l_dt + 0.5f );

            if (l_comp) {
                return -logf( curand_uniform(l_state) )/ampmean - comp;
            } else {
                return -logf( curand_uniform(l_state) )/ampmean;
            }
        } else {
            npcd = pcd - 1;
            if (l_comp) {
                return -comp;
            } else {
                return 0.0f;
            }
        }
    } else {
        return 0.0f;
    }
}

__device__ float regular_jump(float l_lambda, float l_Dp, int l_comp, float l_dt, curandState *l_state)
{
    if (l_Dp != 0.0f) {
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
}

__device__ void predcorr(float &corrl_x1, float l_x1, float &corrl_x2, float l_x2, float &corrl_w, float l_w, int &npcd, int pcd, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_alpha, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, float l_dt)
/* simplified weak order 2.0 adapted predictor-corrector scheme
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 503, p. 532 )
*/
{
    float l_x1t, l_x1tt, l_x2t, l_x2tt, l_wt, l_wtt, predl_x1, predl_x2, predl_w;

    l_x1t = drift(l_x1, l_w, l_amp, l_force) - l_alpha*sinf(l_x2);
    l_x2t = -sinf(l_x2) + l_alpha*drift(l_x1, l_w, l_amp, l_force);
    l_wt = l_omega;

    predl_x1 = l_x1 + l_x1t*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_x2 = l_x2 + l_x2t*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + l_wt*l_dt;

    l_x1tt = drift(predl_x1, predl_w, l_amp, l_force) - l_alpha*sinf(predl_x2);
    l_x2tt = -sinf(predl_x2) + l_alpha*drift(predl_x1, predl_w, l_amp, l_force);
    l_wtt = l_omega;

    predl_x1 = l_x1 + 0.5f*(l_x1t + l_x1tt)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_x2 = l_x2 + 0.5f*(l_x2t + l_x2tt)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    predl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;

    l_x1tt = drift(predl_x1, predl_w, l_amp, l_force) - l_alpha*sinf(predl_x2);
    l_x2tt = -sinf(predl_x2) + l_alpha*drift(predl_x1, predl_w, l_amp, l_force);
    l_wtt = l_omega;

    corrl_x1 = l_x1 + 0.5f*(l_x1t + l_x1tt)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state) + adapted_jump(npcd, pcd, l_lambda, l_Dp, l_comp, l_dt, l_state);
    corrl_x2 = l_x2 + 0.5f*(l_x2t + l_x2tt)*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    corrl_w = l_w + 0.5f*(l_wt + l_wtt)*l_dt;
}

__device__ void eulermaruyama(float &nl_x1, float l_x1, float &nl_x2, float l_x2, float &nl_w, float l_w, curandState *l_state, \
                         float l_amp, float l_omega, float l_force, float l_alpha, float l_Dg, int l_2ndorder, float l_Dp, float l_lambda, int l_comp, float l_dt)
/* simplified weak order 1.0 regular euler-maruyama scheme 
( see E. Platen, N. Bruti-Liberati; Numerical Solution of Stochastic Differential Equations with Jumps in Finance; Springer 2010; p. 508, 
  C. Kim, E. Lee, P. Talkner, and P.Hanggi; Phys. Rev. E 76; 011109; 2007 ) 
*/ 
{
    float l_x1t, l_x2t, l_wt;

    l_x1t = l_x1 + ( drift(l_x1, l_w, l_amp, l_force) - l_alpha*sinf(l_x2) )*l_dt + diffusion(l_Dg, l_dt, l_2ndorder, l_state) 
                 + regular_jump(l_lambda, l_Dp, l_comp, l_dt, l_state);
    l_x2t = l_x2 - sinf(l_x2)*l_dt + l_alpha*( drift(l_x1, l_w, l_amp, l_force)*l_dt + regular_jump(l_lambda, l_Dp, l_comp, l_dt, l_state) )
                 + diffusion(l_Dg, l_dt, l_2ndorder, l_state);
    l_wt = l_w + l_omega*l_dt;

    nl_x1 = l_x1t;
    nl_x2 = l_x2t;
    nl_w = l_wt;
}

__device__ void fold(float &nx, float x, float y, float &nfc, float fc)
//reduce periodic variable to the base domain
{
    nx = x - floor(x/y)*y;
    nfc = fc + floor(x/y)*y;
}

__global__ void run_moments(float *d_x1, float *d_x2, float *d_w, float *d_x1b, float *d_x2b, float *d_dx, curandState *d_states)
//actual moments kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x1, l_x2, l_w, l_x1b, l_x2b, l_dx; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x1 = d_x1[idx];
    l_x2 = d_x2[idx];
    l_w = d_w[idx];
    l_x1b = d_x1b[idx];
    l_x2b = d_x2b[idx];
    l_state = d_states[idx];

    float l_amp, l_omega, l_force, l_alpha, l_Dg, l_Dp, l_lambda;
    int l_comp;

    l_amp = d_amp;
    l_omega = d_omega;
    l_force = d_force;
    l_alpha = d_alpha;
    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;

    //run simulation for multiple values of the system parameters
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
            l_alpha = l_dx;
            break;
        case 'D':
            l_Dg = l_dx;
            break;
        case 'p':
            l_Dp = l_dx;
            break;
        case 'l':
            l_lambda = l_dx;
            break;
    }

    //step size & number of steps
    float l_dt;
    long l_steps, l_trigger, i;

    l_dt = 2.0f*PI/l_omega/d_spp; 
    l_steps = d_steps;
    l_trigger = d_trigger;

    //counters for folding
    float x1fc, x2fc, wfc;
    
    x1fc = 0.0f;
    x2fc = 0.0f;
    wfc = 0.0f;

    int l_2ndorder, pcd;

    l_2ndorder = d_2ndorder;

    if (l_2ndorder) {
        //jump countdown
        pcd = (int) floor( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );
    }
    
    for (i = 0; i < l_steps; i++) {

        //algorithm
        if (l_2ndorder) {
            predcorr(l_x1, l_x1, l_x2, l_x2, l_w, l_w, pcd, pcd, &l_state, l_amp, l_omega, l_force, l_alpha, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, l_dt);
        } else {
            eulermaruyama(l_x1, l_x1, l_x2, l_x2, l_w, l_w, &l_state, l_amp, l_omega, l_force, l_alpha, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, l_dt);
        }
        
        //fold path parameters
        if ( fabs(l_x1) > (2.0f*PI) ) {
            fold(l_x1, l_x1, (2.0f*PI), x1fc, x1fc);
        }

        if ( fabs(l_x2) > (2.0f*PI) ) {
            fold(l_x2, l_x2, (2.0f*PI), x2fc, x2fc);
        }

        if ( l_w > (2.0f*PI) ) {
            fold(l_w, l_w, (2.0f*PI), wfc, wfc);
        }

        if (i == l_trigger) {
            l_x1b = l_x1 + x1fc;
            l_x2b = l_x2 + x2fc;
        }

    }

    //write back path parameters to the global memory
    d_x1[idx] = l_x1 + x1fc;
    d_x2[idx] = l_x2 + x2fc;
    d_w[idx] = l_w;
    d_x1b[idx] = l_x1b;
    d_x2b[idx] = l_x2b;
    d_states[idx] = l_state;
}

__global__ void run_traj(float *d_x1, float *d_x2, float *d_w, curandState *d_states)
//actual trajectory kernel
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    float l_x1, l_x2, l_w; 
    curandState l_state;

    //cache path and model parameters in local variables
    l_x1 = d_x1[idx];
    l_x2 = d_x2[idx];
    l_w = d_w[idx];
    l_state = d_states[idx];

    float l_amp, l_omega, l_force, l_alpha, l_Dg, l_Dp, l_lambda;
    int l_comp;

    l_amp = d_amp;
    l_omega = d_omega;
    l_force = d_force;
    l_alpha = d_alpha;
    l_Dg = d_Dg;
    l_Dp = d_Dp;
    l_lambda = d_lambda;
    l_comp = d_comp;

    //step size & number of steps
    float l_dt;
    long l_steps, i;

    l_dt = 2.0f*PI/l_omega/d_spp; 
    l_steps = d_steps;

    //counters for folding
    float x1fc, x2fc, wfc;
    
    x1fc = 0.0f;
    x2fc = 0.0f;
    wfc = 0.0f;

    int l_2ndorder, pcd;

    l_2ndorder = d_2ndorder;

    if (l_2ndorder) {
        //jump countdown
        pcd = (int) floor( -logf( curand_uniform(&l_state) )/l_lambda/l_dt + 0.5f );
    }
    
    for (i = 0; i < l_steps; i++) {

        //algorithm
        if (l_2ndorder) {
            predcorr(l_x1, l_x1, l_x2, l_x2, l_w, l_w, pcd, pcd, &l_state, l_amp, l_omega, l_force, l_alpha, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, l_dt);
        } else {
            eulermaruyama(l_x1, l_x1, l_x2, l_x2, l_w, l_w, &l_state, l_amp, l_omega, l_force, l_alpha, l_Dg, l_2ndorder, l_Dp, l_lambda, l_comp, l_dt);
        }
        
        //fold path parameters
        if ( fabs(l_x1) > (2.0f*PI) ) {
            fold(l_x1, l_x1, (2.0f*PI), x1fc, x1fc);
        }

        if ( fabs(l_x2) > (2.0f*PI) ) {
            fold(l_x2, l_x2, (2.0f*PI), x2fc, x2fc);
        }

        if ( l_w > (2.0f*PI) ) {
            fold(l_w, l_w, (2.0f*PI), wfc, wfc);
        }

    }

    //write back path parameters to the global memory
    d_x1[idx] = l_x1 + x1fc;
    d_x2[idx] = l_x2 + x2fc;
    d_w[idx] = l_w;
    d_states[idx] = l_state;
}

void prepare()
//prepare simulation
{
    //grid size
    h_paths = (h_paths/h_block)*h_block;
    h_threads = h_paths;

    if (h_moments) h_threads *= h_points;

    h_grid = h_threads/h_block;

    //number of steps
    if (h_traj) {
        h_steps = h_spp;
    } else {
        h_steps = h_periods*h_spp;
    }
    cudaMemcpyToSymbol(d_steps, &h_steps, sizeof(long));
     
    //host memory allocation
    size_f = h_threads*sizeof(float);
    size_ui = h_threads*sizeof(unsigned int);
    size_p = h_points*sizeof(float);

    h_x1 = (float*)malloc(size_f);
    h_x2 = (float*)malloc(size_f);
    h_w = (float*)malloc(size_f);
    h_seeds = (unsigned int*)malloc(size_ui);

    //create & initialize host rng
    curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    curandGenerate(gen, h_seeds, h_threads);
 
    //device memory allocation
    cudaMalloc((void**)&d_x1, size_f);
    cudaMalloc((void**)&d_x2, size_f);
    cudaMalloc((void**)&d_w, size_f);
    cudaMalloc((void**)&d_seeds, size_ui);
    cudaMalloc((void**)&d_states, h_threads*sizeof(curandState));

    //copy seeds from host to device
    cudaMemcpy(d_seeds, h_seeds, size_ui, cudaMemcpyHostToDevice);

    //initialization of device rng
    init_dev_rng<<<h_grid, h_block>>>(d_seeds, d_states);

    free(h_seeds);
    cudaFree(d_seeds);

    //moments specific requirements
    if (h_moments) {
        h_trigger = h_steps*h_trans;
        cudaMemcpyToSymbol(d_trigger, &h_trigger, sizeof(long));

        h_x1b = (float*)malloc(size_f);
        h_x2b = (float*)malloc(size_f);
        h_dx = (float*)malloc(size_p);

        float dxtmp = h_beginx;
        float dxstep = (h_endx - h_beginx)/h_points;

        long i;
        
        //set domainx
        for (i = 0; i < h_points; i++) {
            if (h_logx) {
                h_dx[i] = pow(10.0f, dxtmp);
            } else {
                h_dx[i] = dxtmp;
            }
            dxtmp += dxstep;
        }
        
        cudaMalloc((void**)&d_x1b, size_f);
        cudaMalloc((void**)&d_x2b, size_f);
        cudaMalloc((void**)&d_dx, size_p);
    
        cudaMemcpy(d_dx, h_dx, size_p, cudaMemcpyHostToDevice);
    }
}

void copy_to_dev()
{
    cudaMemcpy(d_x1, h_x1, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size_f, cudaMemcpyHostToDevice);
    if (h_moments) {
        cudaMemcpy(d_x1b, h_x1b, size_f, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x2b, h_x2b, size_f, cudaMemcpyHostToDevice);
    }
}

void copy_from_dev()
{
    cudaMemcpy(h_x1, d_x1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d_w, size_f, cudaMemcpyDeviceToHost);
    if (h_moments) {
        cudaMemcpy(h_x1b, d_x1b, size_f, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_x2b, d_x2b, size_f, cudaMemcpyDeviceToHost);
    }
}

void initial_conditions()
//set initial conditions for path parameters
{
    curandGenerateUniform(gen, h_x1, h_threads);
    curandGenerateUniform(gen, h_x2, h_threads);
    curandGenerateUniform(gen, h_w, h_threads);

    long i;

    for (i = 0; i < h_threads; i++) {
        h_x1[i] *= 2.0f*PI; //x1 in (0,2\pi]
        h_x2[i] *= 2.0f*PI; //x2 in (0,2\pi]
        h_w[i] *= 2.0f*PI; //w in (0,2\pi]
    }

    if (h_moments) {
        memset(h_x1b, 0, size_f);
        memset(h_x2b, 0, size_f);
    }
    
    copy_to_dev();
}

void moments(float *av, float *av2)
//calculate the first two moments of v
{
    float sx1, sx1b, sx2, sx2b;
    int i, j;

    cudaMemcpy(h_x1, d_x1, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x1b, d_x1b, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2, d_x2, size_f, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x2b, d_x2b, size_f, cudaMemcpyDeviceToHost);

    for (j = 0; j < h_points; j++) {
        sx1 = 0.0f;
        sx1b = 0.0f;
        sx2 = 0.0f;
        sx2b = 0.0f;

        for (i = 0; i < h_paths; i++) {
            sx1 += h_x1[j*h_paths + i];
            sx1b += h_x1b[j*h_paths + i];
            sx2 += h_x2[j*h_paths + i];
            sx2b += h_x2b[j*h_paths + i];
        }

        av[j] = (sx1 - sx1b)/( (1.0f - h_trans)*h_periods*2.0f*PI/h_omega )/h_paths;
        av2[j] = (sx2 - sx2b)/( (1.0f - h_trans)*h_periods*2.0f*PI/h_omega )/h_paths;
    }
}

void ensemble_average(float *h_x1, float *h_x2, float &sx1, float &sx2)
//calculate ensemble average
{
    int i;

    sx1 = 0.0f;
    sx2 = 0.0f;

    for (i = 0; i < h_threads; i++) {
        sx1 += h_x1[i];
        sx2 += h_x2[i];
    }

    sx1 /= h_threads;
    sx2 /= h_threads;
}

void finish()
//free memory
{

    free(h_x1);
    free(h_x2);
    free(h_w);
    
    curandDestroyGenerator(gen);
    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_w);
    cudaFree(d_states);
    
    if (h_moments) {
        free(h_x1b);
        free(h_x2b);
        free(h_dx);

        cudaFree(d_x1b);
        cudaFree(d_x2b);
        cudaFree(d_dx);
    }
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
    
    //asymptotic long time average velocity <<v>> and <<v^2>>
    if (h_moments) {
        float *av, *av2;
        int i;

        av = (float*)malloc(size_p);
        av2 = (float*)malloc(size_p);

        if ( !strcmp(h_domain, "1d") ) {
            run_moments<<<h_grid, h_block>>>(d_x1, d_x2, d_w, d_x1b, d_x2b, d_dx, d_states);
            moments(av, av2);

            printf("#%c <<v1>> <<v2>>\n", h_domainx);
            for (i = 0; i < h_points; i++) {
                printf("%e %e %e\n", h_dx[i], av[i], av2[i]);
            }

        } else {
            float h_dy, dytmp, dystep;
            int j;
            
            dytmp = h_beginy;
            dystep = (h_endy - h_beginy)/h_points;
            
            printf("#%c %c <<v1>> <<v2>>\n", h_domainx, h_domainy);
            
            for (i = 0; i < h_points; i++) {
                if (h_logy) {
                    h_dy = pow(10.0f, dytmp);
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
                        cudaMemcpyToSymbol(d_alpha, &h_dy, sizeof(float));
                        break;
                    case 'D':
                        cudaMemcpyToSymbol(d_Dg, &h_dy, sizeof(float));
                        break;
                    case 'p':
                        cudaMemcpyToSymbol(d_Dp, &h_dy, sizeof(float));
                        break;
                    case 'l':
                        cudaMemcpyToSymbol(d_lambda, &h_dy, sizeof(float));
                        break;
                }

                run_moments<<<h_grid, h_block>>>(d_x1, d_x2, d_w, d_x1b, d_x2b, d_dx, d_states);
                moments(av, av2);
                
                for (j = 0; j < h_points; j++) {
                    printf("%e %e %e %e\n", h_dx[j], h_dy, av[j], av2[j]);
                }

                //blank line for plotting purposes
                printf("\n");

                initial_conditions();

                dytmp += dystep;
            }
        }

        free(av);
        free(av2);
    }

    //ensemble averaged trajectory <x>(t) 
    if (h_traj) {
        float t, sx1, sx2;
        int i;

        for (i = 0; i < h_periods; i++) {
            run_traj<<<h_grid, h_block>>>(d_x1, d_x2, d_w, d_states);
            copy_from_dev();
            t = i*2.0f*PI/h_omega;
            ensemble_average(h_x1, h_x2, sx1, sx2);
            printf("%e %e %e\n", t, sx1, sx2);
        }
    }

    //the final position of all paths
    if (h_hist) {
        int i;

        run_traj<<<h_grid, h_block>>>(d_x1, d_x2, d_w, d_states);
        copy_from_dev();
        
        for (i = 0; i < h_threads; i++) {
            printf("%e %e\n", h_x1[i], h_x2[i]);
        }
    }

    finish();

    return 0;
}
