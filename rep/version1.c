/* Jacobi for 1-D Laplacian */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char **argv)

{

  int size=10000, nit = 100000;
  int nprocs, i, iam, n, left, right, count;
  int tag1, tag2, k;
  double *h, *hnew;
  double h0=1.0, hL=0.0,q=0;
  double time1, time2, mflops, total_time,K1,K2,K3;
  double a = 0.007, b= -0.07, c=0.2, C0 = 1.2, C1 = 2.1;
  double errsq, error, hanal, sumsqerr;
  double Qs=0, Qt, Qfinal,Q,f,t; 
  MPI_Status status;

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

/* initialize h */
  for (i=0; i <= n+1; i++) h[i] = 0.0;
  if (iam == 0) h[0] = h0;
  if (iam == nprocs - 1) h[n+1] = hL;

  if(iam == 0) left = MPI_PROC_NULL;
       else left = iam - 1;
  if(iam == nprocs - i) right = MPI_PROC_NULL;
       else right = iam + 1;


/* iterate */
  for (k=1; k <= nit; k++)
      {
/* if i am not processor nprocs-1 send to right neighbor */
       if (iam != nprocs - 1) MPI_Send
           (&h[n], count, MPI_DOUBLE, right, tag1, MPI_COMM_WORLD);
/* if i am not processor 0 receive from left neighbor */
          if (iam != 0 )  MPI_Recv 
           (&h[0], count, MPI_DOUBLE, left, tag1, MPI_COMM_WORLD, &status);
/* if i am not processor 0 send to left neighbor */
       if (iam != 0 ) MPI_Send
           (&h[1], count, MPI_DOUBLE, left, tag2, MPI_COMM_WORLD); 
/* if i am not processor nprocs-1 receive from right neighbor */
       if (iam != nprocs - 1) MPI_Recv
           (&h[n+1], count, MPI_DOUBLE, right, tag2, MPI_COMM_WORLD, &status);/* compute interior */
       for (i = 1; i <= n; i++) 
            K1 = 0.007*((iam*n+i-1)/1000)*((iam*n+i-1)/1000) + (-0.07)*((iam*n+i-1)/1000) + 0.2; 
            K2 = 0.007*((iam*n+i)/1000)*((iam*n+i)/1000) + (-0.07)*((iam*n+i)/1000) + 0.2; 
            K3 = 0.007*((iam*n+i+1)/1000)*((iam*n+i+1)/1000)+ (-0.07)*((iam*n+i+1)/1000)+ 0.2; 
            hnew[i] = ((K2+K1)*h[i+1] + (K1+K3)*h[i-1])/(K2+2*K1+K3); 
            Qs = Qs+ (K1 * 100 * (hnew[i+1] - hnew[i])*1000);
    Qt = Qs * (iam*n+i)/(size+1);
    MPI_Reduce(&Qt, &Qs, 1, MPI_DOUBLE, MPI_SUM, 0,
       MPI_COMM_WORLD);
/* update */
       for (i = 1; i <= n; i++) h[i] = hnew[i];
/* preserve boundary values */
       if (iam == 0) h[0] = h0;
       if (iam == nprocs - 1) h[n+1] = hL;
       }

/* end timer */
   time2=MPI_Wtime();

   total_time=time2-time1;
   mflops = 2*nit*size*1e-6/(total_time);


/* check against analytical solution */
   errsq=0;
   for (i=1; i <=n; i++) {
        t = (iam*n+i)/1000;
	f = sqrt(4*a*c - b*b);
	f = (2 / f) * atan((2*a*t + b) / f);

       hanal = C0*f + C1;
       errsq = errsq + (hanal-h[i])*(hanal-h[i]);
       }
     MPI_Reduce(&errsq, &sumsqerr, 1, MPI_DOUBLE, MPI_SUM, 0,
       MPI_COMM_WORLD);
     error = sqrt(sumsqerr)/size; 
/* print results */
   if (iam == 0) {
       printf("Total time  = %g secs\n",total_time);
       printf("Mflops  = %g\n",mflops);
        printf("Error  = %.16f\n", error);
       printf("Leakage FLux = %g\n",Qt);
   }

   MPI_Finalize();
   return (0);
}
