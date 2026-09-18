"""Exact interval certificate for independence via Silverman's height bound.

Sage's silverman_height_bound() documents |h_x(P)-hhat(P)| <= B. This
module uses an upward integer bound on its displayed formula, computes
exact x(2^n P), and encloses every logarithm by rational atanh series.
Strictly positive principal minors prove independence.
"""
from sage.all import QQ, ZZ
from itertools import permutations

def _bit_upper(q):
    q=QQ(q)
    return max(abs(q.numerator()).nbits(),q.denominator().nbits())

def silverman_integer_upper(E):
    """Round every positive log term upward via log(n) < nbits(n).

    The constants 0.961 and 1.07 in Sage's Silverman formula are both
    smaller than 2, and log(twostar)/2 is at most 1/2.
    """
    delta=E.discriminant();j=E.j_invariant();b=E.b2()/12
    mu=QQ(_bit_upper(delta))/12+QQ(abs(j.numerator()).nbits())/12+QQ(abs(b.numerator()).nbits())/2+QQ(1)/2
    return ZZ((2*(QQ(_bit_upper(j))/24+mu+2)).ceil())

def log_interval(H,bits=40,terms=12):
    H=ZZ(H)
    if H<1:raise ValueError('log input must be positive')
    if H==1:return QQ(0),QQ(0)
    scale=ZZ(1)<<bits
    def series(z):
        z=QQ(z);total=QQ(0);power=z
        for k in range(terms):
            total+=2*power/(2*k+1)
            power*=z*z
        tail=2*power/((2*terms+1)*(1-z*z))
        return total,total+tail
    ln2lo,ln2hi=series(QQ(1)/3)
    exponent=H.nbits()-1
    base=ZZ(1)<<exponent
    q=((H-base)*scale)//(H+base)
    zlo=QQ(q)/scale;zhi=QQ(q+1)/scale
    mlo,_=series(zlo);_,mhi=series(zhi)
    return exponent*ln2lo+mlo,exponent*ln2hi+mhi

def _add(a,b):return a[0]+b[0],a[1]+b[1]
def _neg(a):return -a[1],-a[0]
def _sub(a,b):return _add(a,_neg(b))
def _mul(a,b):
    vals=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
    return min(vals),max(vals)
def _div(a,n):return a[0]/n,a[1]/n

def height_interval(P,n,bound):
    Q=(ZZ(1)<<n)*P
    x=Q[0]
    H=max(abs(x.numerator()),x.denominator())
    lo,hi=log_interval(H)
    factor=ZZ(1)<<(2*n)
    return (lo-bound)/factor,(hi+bound)/factor, H.nbits()

def _det_interval(matrix,size):
    total=(QQ(0),QQ(0))
    for perm in permutations(range(size)):
        term=(QQ(1),QQ(1))
        inversions=sum(perm[i]>perm[j] for i in range(size) for j in range(i+1,size))
        for i,j in enumerate(perm):term=_mul(term,matrix[i][j])
        total=_add(total,_neg(term) if inversions%2 else term)
    return total

def certify(E,points,max_doublings=6):
    rank=len(points)
    if not 1<=rank<=4:raise ValueError('requires one to four points')
    bound=silverman_integer_upper(E)
    pair_indices=[(i,j) for i in range(rank) for j in range(i+1,rank)]
    pairs=list(points)+[points[i]+points[j] for i,j in pair_indices]
    for n in range(2,max_doublings+1):
        heights=[];bits=[]
        for P in pairs:
            lo,hi,size=height_interval(P,n,bound)
            heights.append((lo,hi));bits.append(size)
        matrix=[[None]*rank for _ in range(rank)]
        for i in range(rank):matrix[i][i]=heights[i]
        for k,(i,j) in enumerate(pair_indices,rank):
            matrix[i][j]=matrix[j][i]=_div(_sub(_sub(heights[k],heights[i]),heights[j]),2)
        minors=[_det_interval(matrix,size) for size in range(1,rank+1)]
        if all(x[0]>0 for x in minors):
            return {'certified_rank':rank,'doublings':n,'silverman_bound_integer':str(bound),
                    'height_bits':bits,
                    'principal_minor_lower_bounds':[str(x[0]) for x in minors],
                    'principal_minor_upper_bounds':[str(x[1]) for x in minors],
                    'method':'exact rational log intervals, exact x(2^n P), Silverman height difference bound'}
    return {'certified_rank':None,'doublings':max_doublings,'silverman_bound_integer':str(bound),
            'height_bits':bits,'principal_minor_lower_bounds':[str(x[0]) for x in minors],
            'method':'inconclusive interval; increase max_doublings'}
