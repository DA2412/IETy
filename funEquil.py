# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:41:02 2022
A set of functions for computing the magnetic vector potential, fields and flux 
for axisymmetric geometry.
@author: abate
"""
import numpy as np
def fun_Field_Loop(source, points):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %   Computation of Aphi, Br, Bz, Psi for axisymmetric loop current (no
    %   thickness of the loop)
    %
    %   source (sources' geometry) - dictionary including:  
    %     - R: radial distance form the axis of the coil's centre [m] 
    %     - Z: vertical distance from z=0 plane of the coil's centre [m]
    %
    %   points (evaluation points) - dictionary including:  
    %     - RR: array of the radial coordinate of the evaluatin points [m]
    %     - ZZ: array of the vertical coordinate of the evaluatin points [m]
    %
    %   res (results) - structure including:
    %    - a (npt x ncoil)
    %    - br (npt x ncoil)
    %    - bz (npt x ncoil)
    %    - psi (npt x ncoil) [Wb]
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    RR=np.array(points['R']).flatten('F')
    ZZ=np.array(points['Z']).flatten('F')
    npt=len(RR)#points.shape[0]
    ncoil=len(source['R'])
    a=np.zeros([npt,ncoil])
    br=np.zeros([npt,ncoil])
    bz=np.zeros([npt,ncoil])
    psi=np.zeros([npt,ncoil])
    
    for  jj in range(0,ncoil):
        #print("{}/{}".format(jj,ncoil-1))
        r0=source['R'][jj]
        z0=source['Z'][jj]
        kk=np.sqrt(np.divide(4*np.multiply(r0,RR),((r0+RR)**2+(ZZ-z0)**2)))
        #[J1,J2] = ellipke(kk.^2);
        from scipy.special import ellipk,ellipe
        J1=ellipk(kk**2)
        J2 = ellipe(kk**2)

        B = np.multiply((1-kk**2/2),J1)-J2
       
        res_a = np.multiply(np.multiply(np.divide(4e-7,kk),np.sqrt(r0/RR)),B)
        res_psi=  np.multiply(res_a,2*np.pi*RR)

        uno = np.multiply(1e-7*kk,(ZZ-z0))
        due = np.multiply(RR,np.sqrt(r0*RR))
        tre = -J1+ np.divide((r0-RR)**2+(ZZ-z0)**2 , ((r0-RR)**2+(ZZ-z0)**2))
        quattro = J2  
        res_br= np.multiply(np.multiply(np.divide(uno,due),tre),quattro)
        
        uno = np.divide(1e-7*kk,np.sqrt(r0*RR))
        due = np.divide((r0**2-RR**2-(ZZ-z0)**2) , ((r0-RR)**2+(ZZ-z0)**2))
        tre = J1+np.multiply(due,J2)
        res_bz=np.multiply(uno ,tre ) 
             
        a[:,jj]=res_a
        psi[:,jj]=res_psi
        br[:,jj]=res_br
        bz[:,jj]=res_bz
    
    #points on axis (r=0)
    ind_axis=np.argwhere(RR==0)
    a[ind_axis,:]=0
    psi[ind_axis,:]=0
    br[ind_axis,:]=0
    if ind_axis.size!=0:
        for  jj in range(0,ncoil):
            r0=source['R'][jj]
            z0=source['Z'][jj]
            bz[ind_axis,jj]=(4e-7*np.pi*r0**2)/(2*(r0**2+(ZZ[ind_axis]-z0)**2)**1.5)
    
    #%% Output
    #out = {"G_a",a ,"G_br", br, "G_bz",  bz, "G_psi", psi}
    return a,br,bz,psi

from scipy.special import p_roots, roots_legendre

def gaussWeights(n):
    # [x,w] = p_roots(n)
    [x,w] = roots_legendre(n)
    return x,w

def fun_Field_Coil( source, points, NGauss ):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %   Computation of Aphi, Br, Bz, Psi for axisymmetric coil current (thickness
    %   of the coil is taken into account)
    %
    %   source (sources' geometry) - structure including:
    %     - R: radial distance form the axis of the coil's centre [m]
    %     - Z: vertical distance from z=0 plane of the coil's centre [m]
    %     - DR: coil's width [m]
    %     - DZ: coil's height [m]
    %     - turns: number of turns
    %
    %   point (evaluation points) - structure including:
    %     - RR: array of the radial coordinate of the evaluatin points [m]
    %     - ZZ: array of the vertical coordinate of the evaluatin points [m]
    %
    %   Ngauss number of Gauss integration points
    %
    %   res (results) - structure including:
    %    - a (npt x ncoil)
    %    - br (npt x ncoil)
    %    - bz (npt x ncoil)
    %    - psi (npt x ncoil)
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    # %% Sources and points
    points_rr=np.array(points['R']).flatten('F')
    npt = len(points_rr)
    ncoil=len(source['R'])
    source_R=source['R']
    source_Z=source['Z']
    source_DR=source['DR']
    source_DZ=source['DZ']
    #source_turns=source['R']
    
    # Computation for each coil
    res_a=np.zeros([npt,ncoil])
    res_br=np.zeros([npt,ncoil])
    res_bz=np.zeros([npt,ncoil])
    res_psi=np.zeros([npt,ncoil])

    for jj in range(0,ncoil):
        #print("{}/{}".format(jj,ncoil-1))
        r0=source_R[jj]
        z0=source_Z[jj]
        dr=source_DR[jj]
        dz=source_DZ[jj]
        z1=z0-0.5*dz
        z2=z0+0.5*dz
        rr=points_rr
        #% Gauss weights
        [xi,wi] = gaussWeights(NGauss)
        #        % Integration of thin solenoid along r
        aphi=np.zeros(npt)
        br=np.zeros(npt)
        bz=np.zeros(npt)
        for hh in range(0,NGauss):
            r1=r0+0.5*dr*xi[hh]
            atemp,brtemp,bztemp,psitemp=fun_ThinSolenoid(r1,z1,z2,points)#;% calcolato per corrente unitaria
            aphi= aphi+atemp*wi[hh]*0.5#*source_turns(jj);
            br=   br+brtemp*wi[hh]*0.5#*source_turns(jj);
            bz=   bz+bztemp*wi[hh]*0.5#*source_turns(jj);
        res_a[:,jj]=   aphi
        res_br[:,jj]=  br
        res_bz[:,jj]=  bz
        res_psi[:,jj]= aphi*2*np.pi*rr  
    
    return res_a,res_br,res_bz,res_psi
   

def fun_ThinSolenoid(r1,z1,z2,points):
    #%% parameters
    eps=np.finfo(float).eps
    mu0=4*np.pi*1.e-7
    ier=0
    
    RR=np.array(points['R']).flatten('F')
    ZZ=np.array(points['Z']).flatten('F')
    npt=len(RR)#points.shape[0]
    # ncoil=len(source['R'])
    aphi=np.zeros([npt])
    br=np.zeros([npt])
    bz=np.zeros([npt])

    IK=np.zeros([npt])
    IE=np.zeros([npt])
    IP=np.zeros([npt])
    IPmIK=np.zeros([npt])
    IKmIE=np.zeros([npt])
    
    zeta=np.zeros([npt,2])
    J=1/(z2-z1)#%linear current density [A/m] for unit current I [A]
    rr=RR
    
    rho1=rr/r1 #% normalized r
    sqrt_rho1=np.sqrt(rho1)
    sqn=4*np.divide(rho1,np.multiply((rho1+1),(rho1+1)))
    
    zeta[:,0]=(z1-ZZ)/r1# % normalized deltaz1
    zeta[:,1]=(z2-ZZ)/r1#% normalized deltaz2
    
    #%% find points on the axis
    ind_axis=np.argwhere(rho1<eps)
    aphi[ind_axis]=0.
    br[ind_axis]=0.
    bz[ind_axis]=mu0*J/2.*(np.divide(zeta[ind_axis,1],np.sqrt(1.+np.multiply(zeta[ind_axis,1],zeta[ind_axis,1])))-
                           np.divide(zeta[ind_axis,0],np.sqrt(1 +np.multiply(zeta[ind_axis,0],zeta[ind_axis,0]))))
    ind_all=np.asarray([i for i in range(0,len(rho1))])
    ind_not=np.setdiff1d(ind_all,ind_axis)
    
    #%% All the points off the axis
    ind_off=np.setdiff1d(ind_all[ind_not],ind_axis)
    ind_off_1=np.argwhere(sqn[ind_not]>1.-eps)
    ind_off_2=np.setdiff1d(ind_off,ind_off_1)
    
    #% If the calculation points belongs to the thin solenoid    
    if np.any(ind_off_1): #if not empty
        # print("solenoid")
        for ii in range(0,2):
            fk=np.multiply(rho1+1.,rho1+1.)+np.multiply(zeta[:,ii],zeta[:,ii])
            sqk=4.*np.divide(rho1,fk)
            kk=np.sqrt(sqk)
            for hh in range(0,len(ind_off_1)):
                hh_ok=ind_off_1[hh]
                [ik,ie,ikmie]=ellip_ke(sqk[hh_ok])
                IK[hh_ok]=ik
                IE[hh_ok]=ie
                IKmIE[hh_ok]=ikmie

            #order of executing of mathematical operations elementbyelement (./,.*,...) is from left to right
            aphi[ind_off_1]=aphi[ind_off_1]+(-1.)**(ii+1)*np.multiply(np.divide(zeta[ind_off_1,ii],kk[ind_off_1]),IKmIE[ind_off_1])
            br[ind_off_1]=br[ind_off_1]+(-1.)**(ii+1)*np.multiply(np.divide(1.,kk[ind_off_1]),np.multiply(1-0.5*sqk[ind_off_1],IK[ind_off_1])-IE[ind_off_1])
            
            bz[ind_off_1]=bz[ind_off_1]+(-1.)**(ii+1)*np.multiply(np.multiply(zeta[ind_off_1,ii],kk[ind_off_1]),IK[ind_off_1])

        
        aphi[ind_off_1]=aphi[ind_off_1]*2e-7*J*r1
        br[ind_off_1]=br[ind_off_1]*4e-7*J
        bz[ind_off_1]=bz[ind_off_1]*1e-7*np.divide(J,sqrt_rho1[ind_off_1])

#######################################################################    
    #% If the calculation points does not belong to the thin solenoid     
    for ii in range(0,2):
        # print(ii)
        fk=np.multiply(rho1+1,rho1+1)+np.multiply(zeta[:,ii],zeta[:,ii])
        sqk=4.*np.divide(rho1,fk).flatten()
        kk=np.sqrt(sqk)
        for hh in range(0,len(ind_off_2)):
            hh_ok=ind_off_2[hh]
            [ik,ie,ip,ipmik,ipmie]=ellip_kep(sqn[hh_ok],sqk[hh_ok])
            IK[hh_ok]=ik
            IE[hh_ok]=ie
            IP[hh_ok]=ip
            IPmIK[hh_ok]=ipmik
            IKmIE[hh_ok]=ipmie
        aphi[ind_off_2]=aphi[ind_off_2]+(-1)**(ii+1)*np.multiply(np.divide(zeta[ind_off_2,ii],kk[ind_off_2]),np.multiply(np.divide(np.multiply(sqk[ind_off_2],sqn[ind_off_2]-1.),sqn[ind_off_2]),IPmIK[ind_off_2])+IKmIE[ind_off_2])
        br[ind_off_2]=br[ind_off_2]+np.multiply((-1)**(ii+1)/kk[ind_off_2],np.multiply(1-np.multiply(0.5*kk[ind_off_2],kk[ind_off_2]),IK[ind_off_2])-IE[ind_off_2]) 
        bz[ind_off_2]=bz[ind_off_2]+np.multiply(np.multiply((-1)**(ii+1)*zeta[ind_off_2,ii],kk[ind_off_2]),
                  np.multiply(np.divide(-(rho1[ind_off_2]-1.),rho1[ind_off_2]+1.),
                              IP[ind_off_2])+IK[ind_off_2])
        
    aphi[ind_off_2]=np.divide(aphi[ind_off_2]*2e-7*J*r1,sqrt_rho1[ind_off_2])
    br[ind_off_2]=np.divide(br[ind_off_2]*4e-7*J,sqrt_rho1[ind_off_2])
    bz[ind_off_2]=np.divide(bz[ind_off_2]*1e-7*J,sqrt_rho1[ind_off_2])
    psi=np.multiply(aphi*2*np.pi,rr)
    
    #%% Output
    return aphi,br,bz,psi

def ellip_ke(sqk):
      '''
      % complete elliptic integrals of first, second, third type: K(sqk), E(sqk), P(sqn,sqk)
      # %
      # % accuracy: eps
      # %
      # % ier:      errore index:     =0: OK
      # %                             =1: singularity for sqk
      # %                             =2: max iteration exceeded
      # %
      # % Iterative procedure, based on Landen's transformation, described in
      # % M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
      # % Dover Publications", 1965, Chapter 17.6, pag. 598-599
      '''
# %% parameters
      imax=100
      eps=np.finfo(float).eps
      #eps=1.e-15
      pihalf=np.pi/2
# %% main code
      ier=0     
      if sqk>(1.-eps):
          print('*** FATAL ERROR in ellip_ke: sqk>1-eps ***');
          ier=1
      if(sqk <= eps): 
          IK=pihalf
          IE=pihalf    
      aa0=1.
      bb0=np.sqrt(1.-sqk)
      cc0=np.sqrt(sqk)
      sumc=cc0*cc0
      ii=0   
      while cc0>eps:
          ii=ii+1
          aa=0.5*(aa0+bb0)
          bb=np.sqrt(aa0*bb0)
          cc=0.5*(aa0-bb0)
          aa0=aa
          bb0=bb
          cc0=cc
          sumc=sumc+2**ii*cc*cc           
          # Controllo superamento numero massimo di iterazioni
          if(ii>imax):
              print(['*** ii>imax=.format{%2.0}',imax,' in ellipkep ***']);
              print(['*** Too many iterations ***']);
              ier=2
        
# 17.6.3 kkk=pihalf/aa   
      IK=pihalf/(aa);
# 17.6.4 eee=kkk*(1-sumc/2)
      IE=IK*(1.-sumc*0.5)
      IKmIE=IK*sumc*0.5
      
      return IK,IE,IKmIE
  

def ellip_kep(sqn_hh, sqk_hh):
    '''
    % complete elliptic integrals of first, second, third type: K(sqk), E(sqk), P(sqn,sqk)
    %
    % accuracy: eps
    %
    % ier:      errore index:     =0: OK
    %                             =1: singularity for sqk
    %                             =2: singularity for sqn
    %                             =3: max iteration exceeded
    %
    % Iterative procedure, based on Landen's transformation, described in
    % M. Abramowitz and I.A. Stegun, "Handbook of Mathematical Functions",
    % Dover Publications", 1965, Chapter 17.6, pag. 598-599
    '''
    #%% parameters
    #eps=1.d-15;
    imax = 100
    eps = np.finfo(float).eps
    pihalf = np.pi/2
    #%% main code
    ier = 0
    if sqk_hh > (1.-eps):
     print('*** FATAL ERROR in ellip_ke: sqk>1-eps ***')
     ier = 1
    if sqn_hh > (1.-eps):
     print('*** FATAL ERROR in ellip_ke: sqk>1-eps ***')
     ier = 2
    
    if(sqk_hh <= eps):
     IK = pihalf
     IE = pihalf
     IP = pihalf
    
    aa0 = 1.
    bb0 = np.sqrt(1.-sqk_hh)
    cc0 = np.sqrt(sqk_hh)
    dd0 = (1.-sqn_hh)/bb0
    ee0 = sqn_hh/(1.-sqn_hh)
    ff0 = 0
    sumc = cc0*cc0
    
    ii = 0
    while ((cc0 > eps)or(dd0-1. > eps)):
        ii = ii+1
        aa = 0.5*(aa0+bb0)
        bb = np.sqrt(aa0*bb0)
        cc = 0.5*(aa0-bb0)
        dd = bb/(4.*aa)*(2.+dd0+1./dd0)
        ee = (dd0*ee0+ff0)/(1.+dd0)
        ff = 0.5*(ee0+ff0)
        
        aa0 = aa
        bb0 = bb
        cc0 = cc
        dd0 = dd
        ee0 = ee
        ff0 = ff
        sumc = sumc+2**ii*cc*cc
        if(ii > imax):
            print(['*** ii>imax=.format{%2.0}', imax, ' in ellipkep ***'])
            print(['*** Too many iterations ***'])
            ier = 3
        
    #17.6.3 kkk=pihalf/aa
    IK = pihalf/(aa)
    # 17.6.4 eee=kkk*(1-sumc/2)
    IE = IK*(1.-sumc*0.5)
    IKmIE = IK*sumc*0.5
    #  ppp=kkk*(1+ff)
    IP = IK*(1+ff)
    #  pmk=kkk*ff)
    IPmIK = IK*ff
    return IK, IE, IP, IPmIK, IKmIE