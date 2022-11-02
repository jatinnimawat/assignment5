/* Jacobi for 1-D Laplacian */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define TRUE 1
#define FALSE 0

int main(int argc, char **argv)

{

  int size=10000, nit = 100000;
  int nprocs, i, iam, n, left, right, count,Q;
  int tag1, tag2, k;
  int ndims=1;
  int dims[ndims], isoperiodic[ndims], reorder;
  double *h, *hnew;
  double h0=1.0, hL=0.0,q=0;
  double time1, time2, mflops, total_time,K1,K2,K3;
  double errsq, error, hanal, sumsqerr;
  MPI_Status status;
  MPI_Comm comm1d;

/* initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &iam);
  

/* start timer */
  time1=MPI_Wtime();

/* assume size is perfectly divisible by nprocs */
  n = size/nprocs;

/* allocate arrays */
  h = (double *) malloc((n+2)*sizeof(double));
  hnew = (double *) malloc((n+2)*sizeof(double));

/* initialize message passing parameters */
  count=1; tag1=1; tag2=2;
  left=iam-1; right=iam+1;

/* topology operations */
  isoperiodic[0]=FALSE;
  reorder=TRUE;
  dims[0]=nprocs;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, isoperiodic, reorder, &comm1d);
  MPI_Comm_rank(comm1d, &iam);
  MPI_Cart_shift(comm1d, 0, 1, &left, &right);

/* initialize h */
  for (i=0; i <= n+1; i++) h[i] = 0.0;
  if (iam == 0) h[0] = h0;
  if (iam == nprocs - 1) h[n+1] = hL;

/* iterate */
  for (k=1; k <= nit; k++)
      {
/* receive into right ghost buffer */
  MPI_Sendrecv(&h[1], count, MPI_DOUBLE, left, tag1, &h[n+1], count, MPI_DOUBLE,
       right, tag1, comm1d, &status);
/* send to right neighbor */
  MPI_Sendrecv(&h[n], count, MPI_DOUBLE, right, tag2, &h[0], count, MPI_DOUBLE,
       left, tag2, comm1d, &status);
/* compute interior */
       for (i = 1; i <= n; i++) K1 = 0.007*i*i + (-0.07)*i + 0.2; K2 = 0.007*(i+1)*(i+1) + (-0.07)*(i+1) + 0.2; K3 = 0.007*(i-1)*(i-1) + (-0.07)*(i-1) + 0.2; 
               hnew[i] = ((K2+K1)*h[i+1] + (K1+K3)*h[i-1])/(K2+2*K1+K3); Q = q + (K1 * 100 * (hnew[i+1] - hnew[i]));
/* compute end points */
      hnew[1] = (h[0]+h[2])/2.0;
      hnew[n] = (h[n-1]+h[n+1])/2.0;
/* update */
       for (i = 1; i <= n; i++) h[i] = hnew[i];
/* preserve boundary values */
       if (iam == 0) h[0] = h0;
       if (iam == nprocs - 1) h[n+1] = hL;
       }
printf("node: %d, h: %lf\n",iam*n+1,h[1]);
printf("node: %d, h: %lf\n",iam*n+n/2,h[n/2]);
if (iam == nprocs -1) printf("node: %d, h: %lf\n",nprocs*n,h[n]);
/* end timer */
   time2=MPI_Wtime();

   total_time=time2-time1;
   mflops = 2*nit*size*1e-6/(total_time);

/* check against analytical solution */
   errsq=0;
   for (i=1; i <=n; i++) {
       hanal = h0+(hL-h0)*(iam*n+i)/(size+1);
       errsq = errsq + (hanal-h[i])*(hanal-h[i]);
       }
   MPI_Reduce(&errsq, &sumsqerr, 1, MPI_DOUBLE, MPI_SUM, 0,
       MPI_COMM_WORLD);
   error = sqrt(sumsqerr)/size;
/* print results */
   if (iam == 0) {
       printf("Total time  = %g secs\n",total_time);
       printf("Mflops  = %g\n",mflops);
       printf("Error  = %.16f\n",sumsqerr);
       printf("Leakage FLux = %g\n",Q);
   }

   MPI_Finalize();
   return (0);
}
