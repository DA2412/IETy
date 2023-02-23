# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:34:58 2023

@author: abate
"""
import numpy as np

def compute_shape_parameters(Boundary):
    router = Boundary["router"]
    zouter = Boundary["zouter"]
    rinner = Boundary["rinner"]
    zinner = Boundary["zinner"]
    rupper = Boundary["rupper"]
    zupper = Boundary["zupper"]
    rlower = Boundary["rlower"]
    zlower = Boundary["zlower"]
    rgeom = .5*( router + rinner)
    zgeom = .5*(zouter + zinner)
    major_radius =  rgeom
    minor_radius = major_radius- rinner
    upper_elongation = ( zupper - zgeom)/( router - rgeom)
    lower_elongation = ( zgeom- zlower)/( router - rgeom)
    elongation = ( zupper - zlower)/( router - rinner)
    horizontal_elongation = 1/ elongation
    upper_triangularity = ( rgeom - rupper)/( rgeom- rinner)
    lower_triangularity = ( rgeom- rlower)/( rgeom- rinner)
    triangularity = ( rgeom - .5*rupper - .5*rlower)/( rgeom- rinner)
    shape_parameters = {"rgeom":rgeom, "zgeom":zgeom,"major_radius": major_radius,"minor_radius":minor_radius,"upper_elongation":upper_elongation,"lower_elongation":lower_elongation,"elongation":elongation,"horizontal_elongation":horizontal_elongation,"upper_triangularity":upper_triangularity,"lower_triangularity":lower_triangularity,"triangularity":triangularity}
    return shape_parameters 

def computeEllipseBoundary(Boundary,shape_parameters):
    num = 50
    a = shape_parameters["minor_radius"]
    router = Boundary["router"]
    zouter = Boundary["zouter"]
    rinner = Boundary["rinner"]
    zinner = Boundary["zinner"]
    rupper = Boundary["rupper"]
    zupper = Boundary["zupper"]
    rlower = Boundary["rlower"]
    zlower = Boundary["zlower"]
    Rmax = router
    Rmin = rinner
    Zmax = zupper
    Zmin = zlower
    Rzmax = rupper
    Rzmin = rlower
    Zrmax = zouter
    Zrmin = zinner
    #Rgeo = shape_parameters["rgeom"]
    delta_u = shape_parameters["upper_triangularity"]#upper triangularity
    delta_l = shape_parameters["lower_triangularity"]   #lower triangularity
    
    #defining new elongations for each quadrant
    Zoff14 = Zrmax
    Zoff23 = Zrmin
    
    k_first = (Zmax-Zoff14)/a
    k_second = (Zmax-Zoff23)/a
    k_third = (Zoff23-Zmin)/a
    k_fourth = (Zoff14-Zmin)/a
            
    r = np.zeros((4,num))
    z = np.zeros((4,num))
    A = np.zeros(4)
    B = np.zeros(4)
    
    A[0] = a*(1+delta_u) 
    B[0] =  k_first*a
    r[0,:] = np.linspace(Rmax, Rzmax, num)
    z[0,:] = Zoff14+ B[0]*(1-((r[0,:]-Rzmax)/ A[0])**2)**0.5
    
    r[3,:] = np.linspace(Rzmin, Rmax, num)
    A[3] = a*(1+delta_l)
    B[3] =  k_fourth*a
    z[3,:] = Zoff14- B[3]*(1-((r[3,:]-Rzmin)/ A[3])**2)**0.5
    
    A[1] = a*(1-delta_u) 
    B[1] =  k_second*a
    r[1,:] = np.linspace(Rzmax, Rmin, num)
    z[1,:] = Zoff23+ B[1]*(1-((-r[1,:]+Rzmax)/ A[1])**2)**0.5
    
    A[2] = a*(1-delta_l)
    B[2] =  k_third*a
    r[2,:]=np.linspace(Rmin, Rzmin, num)
    z[2,:] = Zoff23- B[2]*(1-((-r[2,:]+Rzmin)/ A[2])**2)**0.5
    
    r_ellipse=np.real(r)
    z_ellipse=np.real(z)
    k = [k_first,k_second,k_third,k_fourth]
    return r_ellipse,z_ellipse,k,A,B


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj
def intersections(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

def computeSquareness(Boundary,R_boundary_ref,Z_boundary_ref):
    router = Boundary["router"]
    zouter = Boundary["zouter"]
    rinner = Boundary["rinner"]
    zinner = Boundary["zinner"]
    rupper = Boundary["rupper"]
    zupper = Boundary["zupper"]
    rlower = Boundary["rlower"]
    zlower = Boundary["zlower"]
    Rmax = router
    Rmin = rinner
    Zmax = zupper
    Zmin = zlower
    Rzmax = rupper
    Rzmin = rlower
    Zrmax = zouter
    Zrmin = zinner
    Zoff14 = Zrmax
    Zoff23 = Zrmin
    shape_parameters = compute_shape_parameters(Boundary)
    r_ellipse,z_ellipse,k,A,B = computeEllipseBoundary(Boundary,shape_parameters)
    num = r_ellipse.shape[1]
    
    E = np.zeros([4,2]) #extremal quadrant point matrix
    O = np.zeros([4,2])   #center quadrant point matrix
    C = np.zeros([4,2])   #intersection point between diagonal and ellipse matrix
    D = np.zeros([4,2])   #intersection point between diagonal and real shape matrix
    csi = np.zeros([4,1]) 
    Rdiag = np.zeros((4,num)) #R diagonal coordinates
    Zdiag = np.zeros((4,num)) #Z diagonal coordinates
    
    E[0,:] = [Rmax,Zmax]             #first quadrant
    O[0,:] = [Rzmax,Zoff14]
    Rdiag[0,:] = np.linspace(O[0,0],Rmax,num)
    Zdiag[0,:] = O[0,1]+(E[0,1]-O[0,1])/(E[0,0]-O[0,0])*(Rdiag[0,:]-O[0,0])
    [RC,ZC] = intersections(Rdiag[0,:],Zdiag[0,:],r_ellipse[0,:],z_ellipse[0,:])  #calculation of intersections between ellipse and diagonal
    C[0,0] = RC
    C[0,1] = ZC
    [RD,ZD] = intersections(Rdiag[0,:],Zdiag[0,:],R_boundary_ref,Z_boundary_ref) #calculation on fintersections between real shape and diagonal
    D[0,0] = RD
    D[0,1] = ZD
    
    E[1,:] = [Rmin,Zmax]   #second quadrant
    O[1,:] = [Rzmax,Zoff23]
    Rdiag[1,:] = np.linspace(Rmin,O[1,0],num)
    Zdiag[1,:] = O[1,1]+(E[1,1]-O[1,1])/(E[1,0]-O[1,0])*(Rdiag[1,:]-O[1,0])
    [RC,ZC] = intersections(Rdiag[1,:],Zdiag[1,:],r_ellipse[1,:],z_ellipse[1,:])#calculation of intersections between ellipse and diagonal
    C[1,0] = RC
    C[1,1] = ZC
    [RD,ZD] = intersections(Rdiag[1,:],Zdiag[1,:],R_boundary_ref,Z_boundary_ref)#calculation on fintersections between real shape and diagonal
    D[1,0] = RD
    D[1,1] = ZD
    
    E[2,:] = [Rmin,Zmin]     #third quadrant
    O[2,:] = [Rzmin,Zoff23]
    Rdiag[2,:] = np.linspace(Rmin,O[2,0],num)
    Zdiag[2,:] = O[2,1]+(E[2,1]-O[2,1])/(E[2,0]-O[2,0])*(Rdiag[2,:]-O[2,0]); #defining diagonal between point O and point E
    [RC,ZC] = intersections(Rdiag[2,:],Zdiag[2,:],r_ellipse[2,:],z_ellipse[2,:])  #calculation of intersections between ellipse and diagonal
    C[2,0] = RC
    C[2,1] = ZC
    [RD,ZD] = intersections(Rdiag[2,:],Zdiag[2,:],R_boundary_ref,Z_boundary_ref)#calculation on fintersections between real shape and diagonal
    D[2,0] = RD
    D[2,1] = ZD
    
    E[3,:] = [Rmax,Zmin];   #fourth quadrant
    O[3,:] = [Rzmin,Zoff14];
    Rdiag[3,:] = np.linspace(O[3,0],Rmax,num);
    Zdiag[3,:] = O[3,1]+(E[3,1]-O[3,1])/(E[3,0]-O[3,0])*(Rdiag[3,:]-O[3,0]); #defining diagonal between point O and point E
    [RC,ZC] = intersections(Rdiag[3,:],Zdiag[3,:],r_ellipse[3,:],z_ellipse[3,:]);  #calculation of intersections between ellipse and diagonal
    C[3,0] = RC;
    C[3,1] = ZC;
    [RD,ZD] = intersections(Rdiag[3,:],Zdiag[3,:],R_boundary_ref,Z_boundary_ref); #calculation on fintersections between real shape and diagonal
    D[3,0] = RD;
    D[3,1] = ZD;
    
    #distances calculation for each quadrant
    dOE = np.linalg.norm(O[0,:]-E[0,:])
    dOD = np.linalg.norm(O[0,:]-D[0,:])
    dOC = np.linalg.norm(O[0,:]-C[0,:])
    csi[0] = (dOD-dOC)/(dOE-dOC)
    
    dOE = np.linalg.norm(O[1,:]-E[1,:])
    dOD = np.linalg.norm(O[1,:]-D[1,:])
    dOC = np.linalg.norm(O[1,:]-C[1,:])
    csi[1] = (dOD-dOC)/(dOE-dOC)

    dOE = np.linalg.norm(O[2,:]-E[2,:])
    dOD = np.linalg.norm(O[2,:]-D[2,:])
    dOC = np.linalg.norm(O[2,:]-C[2,:])
    csi[2] = (dOD-dOC)/(dOE-dOC)

    dOE = np.linalg.norm(O[3,:]-E[3,:])
    dOD = np.linalg.norm(O[3,:]-D[3,:])
    dOC = np.linalg.norm(O[3,:]-C[3,:])
    csi[3] = (dOD-dOC)/(dOE-dOC)
    
    return csi


def computeSuperEllipses(Boundary,shape_parameters,csi):
    if csi.shape[0]==0:
        csi = np.zeros([4,1]) 

    r_ellipse,z_ellipse,k,A,B = computeEllipseBoundary(Boundary,shape_parameters)
    num = r_ellipse.shape[1]
    delta_u = shape_parameters["upper_triangularity"]#upper triangularity
    delta_l = shape_parameters["lower_triangularity"]   #lower triangularity
    
    #defining new elongations for each quadrant
    Zoff14 = Boundary["zouter"]
    Zoff23 =  Boundary["zinner"]
    Rgeo = shape_parameters["major_radius"]
    a = shape_parameters["minor_radius"]
    
    epsi = a/Rgeo
    n_exponent = np.zeros([4,1],"float16") 
    x = np.zeros((4,num),dtype="complex")
    y = np.zeros((4,num),dtype="complex")
    zs = np.zeros((4,num),dtype="complex")
  
    x[0,:] =  r_ellipse[0,:]-a*(1/epsi-delta_u)                  
    n_exponent[0] = -np.log(2)/np.log(1/(2**0.5) + csi[0]*(1-1/(2**0.5))) 
    y[0,:] =  B[0]*(1-(x[0,:]/A[0])**n_exponent[0])**(1/n_exponent[0])
    zs[0,:] = y[0,:]+Zoff14
    
    x[1,:] = a*(1/epsi-delta_u)- r_ellipse[1,:]
    n_exponent[1] = -np.log(2)/np.log((0.5)**0.5 + csi[1]*(1-1/(2**0.5)))
    y[1,:] =  B[1]*(1-(x[1,:]/ A[1])**n_exponent[1])**(1/n_exponent[1])
    zs[1,:] = y[1,:]+Zoff23
    
    x[2,:] = a*(1/epsi-delta_l)- r_ellipse[2,:]                  
    n_exponent[2] = -np.log(2)/np.log(1/(2**0.5) + csi[2]*(1-1/(2**0.5)))
    y[2,:] =  B[2]*(1-(x[2,:]/ A[2])**n_exponent[2])**(1/n_exponent[2])
    zs[2,:] = Zoff23-y[2,:]
    
    x[3,:] = -a*(1/epsi-delta_l)+ r_ellipse[3,:]             
    n_exponent[3] = -np.log(2)/np.log(1/(2**0.5) + csi[3]*(1-1/(2**0.5)))
    y[3,:] =  B[3]*(1-(x[3,:]/ A[3])**n_exponent[3])**(1/n_exponent[3])
    zs[3,:] = Zoff14-y[3,:]
    
    zs_smooth = np.real(zs)
    
    x = r_ellipse
    y = zs_smooth
    return x,y


def compute_dual_mesh_line(node, tri):
    mu0 = 4*np.pi*1e-7
    nn=node.shape[0]
    ntri=tri.shape[0]
    mur = 1.
    mur=np.ones(ntri)*mur
    print('costruzione delle matrice con assemblaggio locale')
    kk_nu=np.zeros((nn,nn))
    Area=np.zeros((nn,1))
    Psi=np.zeros((nn,1))
    Iphi=np.zeros((nn,1))

    Ctilde = np.array([[0,-1,-1], [-1,0,1], [1,1,0]])     
    C = np.array([[0,-1,1],[-1,0,1],[-1,1,0]])

    # disegnaMesh=1;
    print('Assemblaggio matrice globale')
    print('computing global matrix')
    nb = np.zeros((ntri,2))
    indn_glob = np.zeros((ntri,3),'int32')
    for ii in range(ntri):
        #print("{}/{}".format(ii,ntri-1))
        #riordino i nodi, dal più basso al più alto
        nodi_loc = np.sort(tri[ii,:])

        # nodi prmali
        n1 = node[nodi_loc[0],:]
        n2 = node[nodi_loc[1],:]
        n3 = node[nodi_loc[2],:] 
    
        #baricentro
        nbar = (n1+n2+n3)/3
        nb[ii,:]=nbar
    
        #lati primali
        e1 = n3 - n2
        e2 = n3 - n1
        e3 = n2 - n1
        
        #nodi duali
        n1d = 0.5*(n2+n3)
        n2d = 0.5*(n1+n3)
        n3d = 0.5*(n1+n2)
    
        #lati duali
        e1d = nbar - n1d
        e2d = n2d- nbar
        e3d = nbar - n3d
      
        #Area cella primale
        Area2=abs(e2[0]*e3[1]-e3[0]*e2[1])
        indn_glob[0,:]=tri[ii,:]

        #calcolo delle matrici Me ed Mf:
        M_mu=np.array([[1,0],[0,1]])/(mur[ii]*mu0)
        Me=np.array([e1d,e2d,e3d])
    
        if (n3[0]!=0):
            Mf=np.array([[-e2[0]/n1d[0], +e1[0]/n2d[0], 0],
            [-e2[1]/n1d[0], +e1[1]/n2d[0], 0]])/(Area2*2*np.pi)
        elif (n1[0]!=0):
            Mf=np.array([[0, -e3[0]/n2d[0], +e2[0]/n3d[0]],
            [0, -e3[1]/n2d[0]], +e2[1]/n3d[0]])/(Area2*2*np.pi)
        else:
            Mf=np.array([-e3[0]/n1d[0], 0, +e1[0]/n3d[0]],
                [-e3[1]/n1d[0], 0, +e1[1]/n3d[1]])/(Area2*2*np.pi)
       
        #print('calcolo della matrice locale kk_loc=Ctilde*(Me*M_mu*Mf)*C-i*omega*Msigma')
        kk_loc=np.dot(Ctilde,np.dot(np.dot(Me,np.dot(M_mu,Mf)),C))
        # assegnazione valori alla matrice globale 
        kk_nu[nodi_loc[0], nodi_loc[0]] = kk_nu[nodi_loc[0], nodi_loc[0]] + kk_loc[0,0]
        kk_nu[nodi_loc[0], nodi_loc[1]] = kk_nu[nodi_loc[0], nodi_loc[1]] + kk_loc[0,1]
        kk_nu[nodi_loc[0], nodi_loc[2]] = kk_nu[nodi_loc[0], nodi_loc[2]] + kk_loc[0,2]
        
        kk_nu[nodi_loc[1], nodi_loc[0]] = kk_nu[nodi_loc[1], nodi_loc[0]] + kk_loc[1,0]
        kk_nu[nodi_loc[1], nodi_loc[1]] = kk_nu[nodi_loc[1], nodi_loc[1]] + kk_loc[1,1]
        kk_nu[nodi_loc[1], nodi_loc[2]] = kk_nu[nodi_loc[1], nodi_loc[2]] + kk_loc[1,2]
        
        kk_nu[nodi_loc[2], nodi_loc[0]] = kk_nu[nodi_loc[2], nodi_loc[0]] + kk_loc[2,0]
        kk_nu[nodi_loc[2], nodi_loc[1]] = kk_nu[nodi_loc[2], nodi_loc[1]] + kk_loc[2,1]
        kk_nu[nodi_loc[2], nodi_loc[2]] = kk_nu[nodi_loc[2], nodi_loc[2]] + kk_loc[2,2]

        #reticolo duale 
        #Area duale 1 (affacciata al nodo 1)
        Atmp1 = abs(np.linalg.det(np.array([e3d, e3/2])))
        Atmp2 = abs(np.linalg.det(np.array([e2d, e2/2])))
        A1d=0.5*(Atmp1+Atmp2)
        #%nb_A1d(nodi_loc(1)) = (tmp1 * Atmp1 + tmp2 * Atmp2)/A1d*2;
    
        #Area duale 2 (affacciata al nodo 2)    
        Atmp1 = abs(np.linalg.det(np.array([e3d, e3/2])))
        Atmp2 = abs(np.linalg.det(np.array([e1d, e1/2])))
        A2d=0.5*(Atmp1+Atmp2)
        #%nb_A2d(nodi_loc(2)) = (tmp1 * Atmp1 + tmp2 * Atmp2)/A2d*2;
            
        #Area duale 3 (affacciata al nodo 3)   
        Atmp1 = abs(np.linalg.det(np.array([e1d, e1/2])))
        Atmp2 = abs(np.linalg.det(np.array([e2d, e2/2])))
        A3d=0.5*(Atmp1+Atmp2)
        #%nb_A3d(nodi_loc(3)) = (tmp1 * Atmp1 + tmp2 * Atmp2)/A3d*2;
            
        Area[nodi_loc[0]]=Area[nodi_loc[0]]+A1d
        Area[nodi_loc[1]]=Area[nodi_loc[1]]+A2d
        Area[nodi_loc[2]]=Area[nodi_loc[2]]+A3d
 
        # plt.triplot(p[:,0],p[:,1],t)
        # plt.plot([nbar[0],n1d[0]],[nbar[1],n1d[1]],'-r')
        # plt.plot([nbar[0],n2d[0]],[nbar[1],n2d[1]],'-r')
        # plt.plot([nbar[0],n3d[0]],[nbar[1],n3d[1]],'-r')
    Area_duale = Area
    return Area_duale, kk_nu