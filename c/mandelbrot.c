#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<complex.h>

int main(int argc, char **argv)
{
	const int exp = 2;
	const double max = 2;
	int maxiter = 1000;
	if(argc>1) maxiter = atoi(argv[1]);

	const double xmin = -2;
	const double xmax = .5;
	const double ymin = -1;
	const double ymax = 1;
	const double dx = .001;
	const double dy = .001;
	const int Nx = (xmax-xmin)/dx;
	const int Ny = (ymax-ymin)/dy;
	double x,y, m;
	int i,j, niter;
	double _Complex z,c;

	for(i=0;i<=Nx;i++)
	{
		x = xmin + i*dx;
		for(j=0;j<=Ny;j++)
		{
			y = ymin + j*dy;
			c = x + y*I;
			z = 0.0 + 0.0*I;
			niter = 0;
			m = cabs(c);
			while(m<max && niter<maxiter)
			{
				z = cpow(z,exp) + c;
				m = cabs(z);
				niter++;
			}
			if(m<max) printf("%f %f\n", x,y );
		}
	}
	return 0;
}
