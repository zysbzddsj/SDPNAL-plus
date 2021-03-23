#include "SDPNALplus.h"
#include<ctime>
using namespace Eigen;
using namespace std;
typedef VectorXd V;
typedef MatrixXd M;
SDPNALplus::SDPNALplus(
    const int       m,
    const int       n,
          M         A[],
    const V         &b,
    const M         &c,
    const double    &eta,
    const double    &delta,
    const double    &sigma,     //10
    const double    &tau,
    const int       max_num
    )
    :
    __m(m),
    __n(n),
    __A(A),
    __b(b),
    __c(c),
    __eta(eta),
    __delta(delta),
    __sigma(sigma),
    __tau(tau),
    __max_num(max_num)
{
    this->doCompute();
    this->solvingBegin();
}
void SDPNALplus::doCompute()
{
    __z = M::Zero(__n, __n);
    __s = M::Zero(__n, __n);
    __x = M::Zero(__n, __n);
    __y = V::Zero(__m);
    ADMMplus(__s, __z, __y, __x);
    __iter_num = 0;
}
void SDPNALplus::solvingBegin()
{
    if(__x.cols() != __n || __x.rows() != __n)
    {
        std::cout<<"x is invalid!"<<std::endl;
        exit(0);
    }
    if(__c.cols() != __n || __c.rows() != __n)
    {
        std::cout<<"c is invalid!"<<std::endl;
        exit(0);
    }
    if(__b.rows() != __m)
    {
        std::cout<<"b is invalid!"<<std::endl;
        exit(0);
    }
}
void SDPNALplus::print()
{
    cout<<"The optimal point is\n"<<__x<<endl;
    cout<<"The objective is\n"<<objective(__x)<<endl;
}
V SDPNALplus::operator_A(const M &x)
{
    V b(__m);
    for(int i=0; i<__m; i++)
    {
        b(i) = ( __A[i].cwiseProduct(x) ).sum();
    }
    return b;
}
M SDPNALplus::operator_Astar(const V &y)
{
    M x=M::Zero(__n,__n);
    for(int i=0; i<__m; i++)
    {
        x = x + y(i)*__A[i];
    }
    return x;
}
M SDPNALplus::operator_A_inv()
{
    //...........?
    int i,j,k;
    Eigen::SparseMatrix<double> smA(__m+__n*(__n-1)/2, __n*__n);
    for (i=0; i < __m; ++i)// 初始化非零元素
    {
        for(j=0; j < __n; ++j)
            for(k=0; k < __n; ++k)
                if( (__A[i])(j,k) != 0 )
                    smA.insert(i, j*__n+k) = (__A[i])(j,k);
    }
    for(i=__m;i<__m+__n*(__n-1)/2;i++)
    {
        for(j=0;j<__n;j++)if(j*(j+1)/2>=i-__m+1)break;
        smA.insert(i,j*__n+(i-__m)-(j*(j-1)/2))=-1;
        smA.insert(i,((i-__m)-(j*(j-1)/2))*__n+j)=1;
    }
    V b=V::Zero(__m+__n*(__n-1)/2);
    for(i=0;i<__m;i++)b(i)=__b(i);
    smA.makeCompressed();// 一个QR分解的实例
    Eigen::SparseQR< Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > linearSolver;// 计算分解
    linearSolver.compute(smA);
    V vecX = linearSolver.solve(b);// 求一个A*x = b
    M x(__n,__n);
    for(int i=0; i<__n; i++)
        for(int j=0; j<__n; j++)
            x(i,j) = vecX(i*__n+j);
    return x;
}
double SDPNALplus::objective(const M &x) const
{
    return ( __c.cwiseProduct(x) ).sum();
}
const M& SDPNALplus::optPoint() const
{
    return __x;
}
M SDPNALplus::projection_K(const M &A)              //..........p(1)<p(2)?
{
    SelfAdjointEigenSolver<M> eigensolver(A);
    M q = eigensolver.eigenvectors();
    V p = eigensolver.eigenvalues();
    M r = p.asDiagonal();
    for(int i=0; i<__n; i++)
        if(r(i,i)<0)
            r(i,i)=0;
    if(q.determinant()<0)q=-q;
    M t=q*r*q.transpose();
    return t;
}
M SDPNALplus::projection_Kstar(const M &A)          //.........projection_K = projection_Kstar?
{
    return projection_K(-A) + A;
}
M SDPNALplus::projection_P(const M &A)
{
    M A1=A;
    for(int i=0; i<__n; i++)
        for(int j=0; j<__n; j++)
            if(A1(i,j)<0)
                A1(i,j)=0;
    return A1;
}
M SDPNALplus::projection_Pstar(const M &A)
{
    return projection_P(-A)+A;
}
M SDPNALplus::compute_Cba(M &Xba,V &y,M &s)
{
    M C1=__c-Xba/__sigma;
    M Zl=projection_Pstar(C1-operator_Astar(y)-s);
    return __c-Zl;
}
V SDPNALplus::operator_vj(V &d, V &y, M &Xba, M &s, const double &epsilon) //compute (Vj+epsilon*I)d
{
    M Cba = compute_Cba(Xba,y,s);
    M A = Xba + __sigma*(operator_Astar(y) - Cba);                            //...........Cba???......A???
    SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
    M q = eigensolver.eigenvectors();
    if(q.determinant()<0)q=-q;
    V p = eigensolver.eigenvalues();
    int i,j;
    double t;
    for(i=0; i<__n/2; i++)
    {
        t = p(i);
        p(i) = p(__n-1-i);
        p(__n-1-i) = t;
    }
    //cout<<p<<endl;
    for(i=0; i<__n; i++)
        for(j=0; j<__n/2; j++)
            {
                t = q(i,j);
                q(i,j) = q(i,__n-1-j);
                q(i,__n-1-j) = t;
            }
    for(i=0; i<__n; i++)
        if(p(i)<=1e-10)
            break;
    int alpha = min(i,__n-1);                 //...............
    if( p(alpha)<=1e-10)
        alpha--;
    M SIG=M::Zero(__n,__n);
    for(i=0; i<__n; i++)
        for(j=0; j<__n; j++)
        {
            if(i<=alpha && j<=alpha)
                SIG(i,j) = 1;
            else if(i<=alpha)
                SIG(i,j) = p(i)/(p(i)-p(j));
            else if(j<=alpha)
                SIG(i,j) = p(j)/(p(j)-p(i));
            else SIG(i,j) = 0;
        }
    M ss = q.transpose();
    V tt=__sigma*operator_A(q*SIG.cwiseProduct(ss*operator_Astar(d)*q)*ss) + epsilon*d;
    return tt;
}
double SDPNALplus::fai(V &y,M &p)
{
    return -__b.dot(y) + __sigma/2*(p.cwiseProduct(p)).sum();
}
V SDPNALplus::differencial_fai(M &t)          //sigma
{
    M p=t*__sigma;
    return operator_A(p) - __b;
}
double SDPNALplus::Psil(V y,M s,M z,M Xba)
{
    MatrixXd p=operator_Astar(y)+s+1/__sigma*Xba-__c+z;
    return -__b.dot(y)+__sigma/2*(p.cwiseProduct(p)).sum();
}
double SDPNALplus::find_mj(V y,V d,double delta,double miu,M Xba,M s)//step 3
{
    double p = 1;
    int k=0;
    M Cba=compute_Cba(Xba,y,s);
    M t=projection_K(operator_Astar(y)+Xba/__sigma-Cba);
    double r = d.dot(differencial_fai(t));
    double t2 = fai(y,t);
    do
    {
        V q=d;
        for(int i=0;i<__m;i++)q(i)=q(i)*p;
        q=q+y;
        M Cba2=compute_Cba(Xba,q,s);
        M tt=projection_K(operator_Astar(q)+Xba/__sigma-Cba);
        double a = fai(q,tt);
        double b = t2+p*miu*r;
        //cout<<a<<' '<<b<<endl;
        if(a<=b)
            break;
        k++;
        p = p*delta;
    }while(p>1e-16);
    //cout<<endl;
    return p;
}
V SDPNALplus::differencial_Psi_part1(V y,M s,M z,M x)
{
    M t=operator_Astar(y)+s-__c+z;
    return operator_A(x+__sigma*(t))-__b;
}
M SDPNALplus::differencial_Psi_part2(V y,M s,M z,M x)
{
    return x+__sigma*(operator_Astar(y)+s-__c+z);
}
bool SDPNALplus::check_A1(VectorXd yn1,VectorXd yn,MatrixXd sn1,MatrixXd sn,double ksi,MatrixXd Xba,MatrixXd z)
{
    VectorXd p=differencial_Psi_part1(yn1,sn1,z,Xba);
    MatrixXd q=differencial_Psi_part2(yn1,sn1,z,Xba);
    double r=p.dot(yn-yn1)+(q.cwiseProduct(sn-sn1)).sum();
    return Psil(yn,sn,z,Xba)<=Psil(yn1,sn1,z,Xba)-ksi/2*abs(r);
}
bool SDPNALplus::check_A2(VectorXd yn1,VectorXd yn,MatrixXd sn1,MatrixXd sn,double ksi,MatrixXd Xba,MatrixXd z)
{
    VectorXd p=differencial_Psi_part1(yn,sn,z,Xba);
    MatrixXd q=sn-projection_K(sn-differencial_Psi_part2(yn,sn,z,Xba));
    double r=ksi*pow(abs(Psil(yn1,sn,z,Xba)-Psil(yn,sn,z,Xba)),0.5);
    double t=pow(p.dot(p)+(q.cwiseProduct(q)).sum(),0.5);
    return t<=r;
}
bool SDPNALplus::check_B3(M t,M x1,M x)
{
    V p=differencial_fai(t);
    cout<<sqrt(p.squaredNorm())<<' '<<__delta/__sigma*sqrt((x1-x).squaredNorm())<<endl;
    return sqrt(p.squaredNorm())<=__delta/__sigma*sqrt((x1-x).squaredNorm());
}
V SDPNALplus::CG(const double &etaj, const int &nj, const double &epsilonj, const V &b, M &A,M &s)         //.....Can CG solve operator(x) = b?
{
    V xn = VectorXd::Zero(__m);
    V rk = b;
    V pk = rk;
    int k = 0;
    double sk, tk;
    V r;
    while(k<nj && sqrt(rk.squaredNorm())>etaj )
    {
        sk = rk.transpose()*rk;
        V vk=operator_vj(pk,__y,A,s,epsilonj);
        sk = sk/( pk.transpose() * vk );
        xn = xn + sk * pk;
        r = rk;
        rk = rk - sk * vk;
        tk = rk.transpose()*rk;
        tk = tk/(r.transpose()*r);
        pk = rk + tk*pk;
        k = k + 1;
    }
    return xn;
}
V SDPNALplus::SNCG(V &y0, M &Xba, double sigma,M &s)//SNCG algorithm
{
    double miu = 0.1;
    double tau1 = 0.1;
    double tau2 = 0.1;
    V yj = y0;
    V x1,x2;
    int nj = 10;               //.......???
    for(int i=1; i<=__n*__n; i++)
    {
        M Cba=compute_Cba(Xba,yj,s);
        M t=projection_K(operator_Astar(yj)+Xba/__sigma-Cba);
        x2 = x1;
        x1 = differencial_fai(t);
        if(i==1)x2=x1;
        cout<<x1.dot(x1)<<endl;
        if(x1.dot(x1)<1e-10)break;
        if(x1.dot(x1)>x2.dot(x2))break;
        double module = sqrt(x1.dot(x1));
        double etaj = min(__eta,pow(module,1+__tau));
        double epsilonj = tau1*min(tau2,module);
        V dj = CG(etaj,nj,epsilonj,-x1,Xba,s);
        double aj = find_mj(yj,dj,__delta,miu,Xba,s);
        //cout<<aj<<endl;
        V w=aj*dj;
        yj = yj + w;
    }
    cout<<endl;
    return yj;
}
void SDPNALplus::MSNCG(V &yn, M &Sn, M &Zn, M Xba,double sigma)
{
    double ksi1=0.5;
    double ksi2=0.5;
    int k=0;
    V yn1;
    M Sn1,Zn1;
    do
    {
        k++;
        yn1 = yn;
        Sn1 = Sn;
        Zn1 = Zn;
        yn = SNCG(yn,Xba,sigma,Sn);
        M t=__c-operator_Astar(yn)-1/__sigma*Xba;
        Sn = projection_K(t-Zn);
        Zn = projection_Pstar(t-Sn);
        if(check_A1(yn1,yn,Sn1,Sn,ksi1,Xba,Zn1)||check_A2(yn1,yn,Sn1,Sn,ksi2,Xba,Zn1))break;
    }while(1);
}
void SDPNALplus::ADMMplus(M &sn, M &zn, V &yn, M &xn)
{
    M AAstar(__m, __m);                              //operator_AAstar_inv(y) = AAstar*y;
    for(int i=0; i<__m; i++)
        for(int j=0; j<__m; j++)
            AAstar(i,j) = ( __A[i].cwiseProduct(__A[j]) ).sum();
    xn = operator_A_inv();
    yn = AAstar.llt().solve( operator_A(__c-sn-zn) );     //AAstar*y = c-sn-zn;
    V yn_half(__m);
    for(int i=0; i<1000; i++)
    {
        //cout<<"i="<<i<<endl;
        sn = projection_K(__c-zn-operator_Astar(yn)-xn/__sigma);
        yn_half = AAstar.llt().solve( operator_A(__c-sn-zn) );
        zn = projection_Pstar(__c-sn-operator_Astar(yn_half)-xn/__sigma);
        yn = AAstar.llt().solve( operator_A(__c-sn-zn) );
        M t=__tau*__sigma*( sn+zn+operator_Astar(yn)-__c );
        //cout<<sqrt((t.cwiseProduct(t)).sum())<<endl;
        if(sqrt((t.cwiseProduct(t)).sum())<1e-6){cout<<i<<endl;break;}
        xn = xn + t;
    }
}
void SDPNALplus::sdpnal_plus()
{
    double rho=1.0;//1.6/0.6
    int k=0;
    M x1;
    M t;
    do
    {
        k++;
        cout<<"k="<<k<<endl;
        //cout<<k<<endl;
        MSNCG(__y,__s,__z,__x,__sigma);
        x1 = __x;
        __sigma = __sigma*rho;
        //__delta = __delta*0.5;
        M w=__sigma*(operator_Astar(__y)+__s+__z-__c);
        if(sqrt((w.cwiseProduct(w)).sum())<1e-10)break;
        __x=__x+w;
        //M Cba=compute_Cba(x1,__y,__s);
        //t=projection_K(operator_Astar(__y)+__z+__s+x1/__sigma-Cba);
        //cout<<k<<endl;
        //if(check_B3(t,x1,__x))break;
    }while(k<__n*__n);
}
int main()
{
    clock_t start=clock();
    ifstream infile("sdpnal_plus_input1.txt");
    int n,m;
    infile>>n>>m;
    int i,j,k;
    double t;
    M A[m];
    V b(m);
    M c(n,n);
    for(i=0;i<n;i++)
        for(j=0;j<n;j++)
           infile>>c(i,j);
    for(i=0;i<m;i++)
    {
        A[i]=MatrixXd::Zero(n,n);
        for(j=0;j<n;j++)
            for(k=0;k<n;k++)
            {
                infile>>t;
                A[i](j,k)=t;
            }
    }
    for(i=0;i<m;i++) infile>>b(i);
    double eta = 0.5;
    double tau = 0.5;
    double delta = 0.5;
    double sigma=10.0;
    SDPNALplus P(m,n,A,b,c,eta,delta,sigma,tau,100);
    P.print();
    //P.sdpnal_plus();
    //P.print();
    infile.close();
    clock_t finish=clock();
    cout<<finish-start<<endl;
}
