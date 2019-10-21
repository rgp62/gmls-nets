import numpy as np

def fourth(x,y,eps):
    r = np.sqrt(np.sum((x - y)**2,1))
    rh = abs(r)
    w = (rh <= eps) * (15./16)*(1-(rh/eps)**2)**2
    return w

def fifth(x,y,eps):
    r = np.sqrt(np.sum((x - y)**2,1))
    rh = abs(r)
    w = (rh <= eps)*(
            -pow(rh-eps,3)*(6.*rh*rh +3.*rh*eps + eps**2)/pow(eps,5)
            )
    return w

def sixth(x,y,eps):
    r = np.sqrt(np.sum((x - y)**2,1))
    rh = abs(r)
    w = (rh <= eps)*pow(1 - rh/ eps, 6)
    return w

def hat(x,y,eps):
    r = np.sqrt(np.sum((x - y)**2,1))
    rh = abs(r)
    w = (rh <= eps)*1.0
    return w
