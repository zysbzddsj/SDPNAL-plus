#include <iostream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#ifndef SDPNALPLUS_H
#define SDPNALPLUS_H
using namespace Eigen;
typedef VectorXd V;
typedef MatrixXd M;
class SDPNALplus
{
private:
	int		__m;
	int		__n;
	M		*__A;
	M       __x;
	V		__b;
	V		__y;
	M		__z;
	M		__s;
	M		__c;
	double	__eta;
	double	__delta;
	double  __sigma;
	double	__tau;
	int		__iter_num;
	int		__max_num;
	double	__error;
	void solvingBegin();
	void doCompute();
	double oneIteration();
public:
	SDPNALplus(
	    const int		m,
		const int		n,
		      M	        A[], 			//*A.....or vector?
		const V			&b,
		const M			&c,
		const double	&eta,
		const double	&delta,
		const double	&sigma,		//10
		const double	&tau,
		const int		max_num
	);
	void print();
	V operator_A(const M &x);
	M operator_Astar(const V &y);
	M operator_A_inv();
	double objective(const M &x) const;
	const M& optPoint() const;
	M projection_K(const M &A);
	M projection_Kstar(const M &A);
	M projection_P(const M &A);
	M projection_Pstar(const M &A);
	M compute_Cba(M &Xba,V &y,M &s);
	V operator_vj(V &d, V &y, M &Xba, M &s, const double &epsilon);
	double fai(V &y, M &p);
	V differencial_fai(M &t);
	double Psil(V y,M S,M Z,M Xba);
	double find_mj(V y,V d,double delta,double miu,M Xba,M s);
    V differencial_Psi_part1(V y,M s,M Z,M x);
    M differencial_Psi_part2(V y,M s,M Z,M x);
    bool check_A1(V yn1,V yn,M sn1,M sn,double ksi,M Xba,M z);
    bool check_A2(V yn1,V yn,M sn1,M sn,double ksi,M Xba,M z);
    bool check_B3(M t,M x1,M x);
	V CG(const double &etaj, const int &nj, const double &epsilonj, const V &b, M &A,M &s);
	V SNCG(V &y0, M &Xba,double sigma,M &s);
	void MSNCG(V &yn, M &Sn, M &Zn, M Xba,double sigma);
	void ADMMplus(M &sn, M &zn, V &yn, M &xn);
	void sdpnal_plus();//solve
	void solve();
};
#endif
