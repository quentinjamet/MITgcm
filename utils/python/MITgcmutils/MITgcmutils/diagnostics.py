import re
import numpy as np

nstats = 5

def readstats(fname):
    '''
    locals,totals,itrs = readstats(fname)

    Read a diagstats text file into record arrays (or dictionaries).

    Parameters
    ----------
    fname : string
        name of diagstats file to read

    Returns
    -------
    locals : record array or dict of arrays
        local statistics, shape (len(itrs), Nr, 5)
    totals : record array or dict of arrays
        column integrals, shape (len(itrs), 5)
    itrs : list of int
        iteration numbers found in the file

    Notes
    -----
    - The 5 columns of the resulting arrays are average, std.dev, min, max and total volume.
    - There is a record (or dictionary key) for each field found in the file.

    '''
    flds = []
    with open(fname) as f:
        for line in f:
            if line.startswith('# end of header'):
                break

            m = re.match(r'^# ([^:]*) *: *(.*)$', line.rstrip())
            if m:
                var,val = m.groups()
                if var.startswith('Fields'):
                    flds = val.split()

        res = dict((fld,[]) for fld in flds)
        itrs = dict((fld,[]) for fld in flds)

        for line in f:
            if line.strip() == '':
                continue

            if line.startswith('# records'):
                break

            m = re.match(r' field : *([^ ]*) *; Iter = *([0-9]*) *; region # *([0-9]*) ; nb\.Lev = *([0-9]*)', line)
            if m:
                fld,itr,reg,nlev = m.groups()
                itrs[fld].append(int(itr))
                tmp = np.zeros((int(nlev)+1,nstats))
                for line in f:
                    if line.startswith(' k'):
                        continue

                    if line.strip() == '':
                        break

                    cols = line.strip().split()
                    k = int(cols[0])
                    tmp[k] = [float(s) for s in cols[1:]]

                res[fld].append(tmp)

            else:
                raise ValueError('readstats: parse error: ' + line)

    try:
        all = np.rec.fromarrays([np.array(res[fld]) for fld in flds], names=flds)
        return all[:,1:],all[:,0],itrs
    except:
        totals = dict((fld,np.array(res[fld])[:,0]) for fld in flds)
        locals = dict((fld,np.array(res[fld])[:,1:]) for fld in flds)
        return locals,totals,itrs


def advec_um(u1, v1, w1, u2, grd, cori=False, metr=False):

    '''
    Compute the 3D advection and advective fluxes
    of zonal momentum.

    Input:
       - u1, v1, w1: three dimensional ADVECTING velocity field
       - u2:         three dimensional ADVECTED  velocity field
       - grd: list of model grid varaibles
    Output:
       - ADVx_Um, ADVy_Um, ADVrE_Um: zonal, meridional and vertical u-momentum flux
       - Um_Advec: u-momentum flux divergence.
                   In MITgcm diagnostics, Um_Advec also includes 
                   Coriolis (Um_Cori) and metric term (mT).
                   They can be computed and added with cori=True (to be done) and metr=True.
    '''

    #-- grid --
    yG  = grd["YG"]
    dxG = grd["DXG"]
    dyG = grd["DYG"]
    drF = grd["DRF"]
    hW  = grd["hFacW"]
    hS  = grd["hFacS"]
    hC  = grd["hFacC"]
    rA  = grd["RAC"]
    rAw = grd["RAW"]
    #
    xA = dyG[np.newaxis, :, :] * drF * hW
    yA = dxG[np.newaxis, :, :] * drF * hS
    maskC = hC * 1.0
    maskC[ np.where(maskC>0 ) ] = 1.0
    #
    recip_rSphere = 1/(6.37e6)
   

    #-- get and check dimensions --
    [nr, ny, nx] = hW.shape
    tmp = nr*ny*nx
    if np.size(u1)!=np.size(v1) or np.size(u1)!=np.size(w1) or np.size(u1)!=tmp or np.size(u2)!= np.size(u1):
      raise ValueError("advec_um: velocity field do not have the same/right dimension")

    #-- zonal advective flux of U (from mom_u_adv_uu.F) --
    #- transport -
    uTrans = u1 * xA
    #- adv flux -
    ADVx_Um = np.zeros([nr, ny, nx])
    ADVx_Um[:, :, :-1] = \
            0.25 *(uTrans[:, :, :-1] + uTrans[:, :, 1:] ) \
                 *(    u2[:, :, :-1] +    u2[:, :, 1:]  )
    
    #-- meridional advective flux of U (from mom_u_adv_vu.F) --
    #- transport -
    vTrans = v1 * yA
    #- adv flux -
    ADVy_Um = np.zeros([nr, ny, nx])
    ADVy_Um[:, 1:, 1:] = \
            0.25 *(vTrans[:, 1:, 1:] + vTrans[:, 1:, :-1] ) \
                 *(    u2[:, 1:, 1:] +     u2[:, :-1, 1:]   )
    
    #-- vertical advective flux of U (from mom_u_adv_wu.F) --
    #- transport -
    # rTransU :: vertical transport (above U point) 
    rTransU = np.zeros([nr, ny, nx])
    rTransU[:, :, 1:] = \
            0.5 * ( w1[:, :, :-1] * rA[np.newaxis, :, :-1] \
                   +w1[:, :, 1: ] * rA[np.newaxis, :, 1: ] )
    #- advective flux -
    # surface layer 
    ADVrE_Um = np.zeros([nr, ny, nx])
    ADVrE_Um[0, :, :] = rTransU[0, :, :] * u2[0, :, :]
    #ADVrE_Um[0, :, :] = 0.0         # rigid lid, for checking
    # interior flux
    ADVrE_Um[1:, :, :] = rTransU[1:, :, :] * \
            0.5 * ( u2[1:, :, :] + u2[:-1, :, :])
    # (linear) Free-surface correction at k>1
    ADVrE_Um[1:, :, 1:] = ADVrE_Um[1:, :, 1:] \
            + 0.25 * (\
              w1[1:, :, 1:] * rA[np.newaxis, :, 1:] *\
                (maskC[1:, :, 1:] - maskC[:-1, :, 1:]) \
             +w1[1:, :, :-1] * rA[np.newaxis, :, :-1] *\
                (maskC[1:, :, :-1] - maskC[:-1, :, :-1]) \
                     ) * u2[1:, :, 1:]
        
    
    #-- flux divergence (from mom_fluxform.F) --
    #- zonal -
    gUx = np.zeros([nr, ny, nx])
    gUx[:, :, 1:] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:, :, 1:] \
            * (ADVx_Um[:, :, 1:] - ADVx_Um[:, :, :-1])
    #- meridional -
    gUy = np.zeros([nr, ny, nx])
    gUy[:, :-1, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:, :-1, :] \
            * (ADVy_Um[:, 1:, :] - ADVy_Um[:, :-1, :])
    #- vertical -
    gUz = np.zeros([nr, ny, nx])
    gUz[:-1, :, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:-1, :, :] \
            * (ADVrE_Um[:-1, :, :]-ADVrE_Um[1:, :, :])
    gUz[-1, :, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[-1, :, :] \
            * (ADVrE_Um[-1, :, :] - 0.0)     #no bottom flux at bottom cell
    #- total -
    Um_Advec = gUx + gUy + gUz


    #-- Coriolis and metric term --
    Um_Cori = np.zeros([nr, ny, nx])
    Um_metr = np.zeros([nr, ny, nx])
    #- Coriolis (from mom_u_coriolis.F) -
    if cori:
      print("Include Coriolis terms in Um_Advec (following MITgcm diagnostic package convention)")
      print("TO BE DONE (07/21/2023)")

    #- metric (from mom_u_metric_sphere.F) -
    if metr:
      print("Include Metric terms in Um_Advec (following MITgcm diagnostic package convention)")
      tanPhiAtU = np.zeros([ny, nx])
      tanPhiAtU[:-1, :] = np.tan( np.deg2rad( 0.5*(yG[:-1, :]+yG[1:, :]) ) )
      #
      Um_metr[:, :-1, 1:] =                         \
        u1[:, :-1, 1:]*recip_rSphere               \
       *0.25*( v1[:, :-1, :-1] + v1[:, :-1, 1:]   \
              +v1[:, 1: , :-1] + v1[:, 1: , 1:] ) \
       *tanPhiAtU[:-1, 1:]


    return Um_Advec, ADVx_Um, ADVy_Um, ADVrE_Um, Um_Cori, Um_metr


def advec_vm(u1, v1, w1, v2, grd, cori=False, metr=False):

    '''
    Compute the 3D advection and advective fluxes
    of meridional momentum.

    Input:
       - u1, v1, w1: three dimensional ADVECTING velocity field
       - v2:         three dimensional ADVECTED  velocity field
       - grd: list of model grid varaibles
    Output:
       - ADVx_Vm, ADVy_Vm, ADVrE_Vm: zonal, meridional and vertical v-momentum flux
       - Vm_Advec: v-momentum flux divergence.
                   In MITgcm diagnostics, Vm_Advec also includes 
                   Coriolis (Vm_Cori) and metric term (mT).
                   They can be computed and added with cori=True (to be donw) and metr=True.
    '''

    #-- grid --
    yG  = grd["YG"]
    dxG = grd["DXG"]
    dyG = grd["DYG"]
    drF = grd["DRF"]
    hW  = grd["hFacW"]
    hS  = grd["hFacS"]
    hC  = grd["hFacC"]
    rA  = grd["RAC"]
    rAs = grd["RAS"]
    #
    xA = dyG[np.newaxis, :, :] * drF * hW
    yA = dxG[np.newaxis, :, :] * drF * hS
    maskC = hC * 1.0
    maskC[ np.where(maskC>0 ) ] = 1.0
    #
    recip_rSphere = 1/(6.37e6)

    #-- get and check dimensions --
    [nr, ny, nx] = hS.shape
    tmp = nr*ny*nx
    if np.size(u1)!=np.size(v1) or np.size(u1)!=np.size(w1) or np.size(u1)!=tmp or np.size(v2)!=np.size(v1):
      raise ValueError("advec_vm: velocity field do not have the same/right dimension")
    #
    
    #-- zonal advective flux of V (mom_v_adv_uv.F) --
    #- transport -
    uTrans = u1 * xA
    #- adv flux -
    ADVx_Vm = np.zeros([nr, ny, nx])
    ADVx_Vm[:, 1:, 1:] = \
        0.25 *(uTrans[:, 1:, 1:] + uTrans[:, :-1, 1:] ) \
             *(    v2[:, 1:, 1:] + v2[:, 1:, :-1]     )

    #-- meridional advective flux of V (from mom_v_adv_vv.F) --
    #- transport -
    vTrans = v1 * yA
    #- adv flux -
    ADVy_Vm = np.zeros([nr, ny, nx])
    ADVy_Vm[:, :-1, :] = \
            0.25 *(vTrans[:, :-1, :] + vTrans[:, 1:, :] ) \
                 *(    v2[:, :-1, :] + v2[:, 1:, :]     )

    #-- vertical advective flux of V (mom_v_adv_wv.F) --
    #- transport -
    # rTransV :: vertical transport (above V point) 
    rTransV = np.zeros([nr, ny, nx])
    rTransV[:, 1:, :] = \
            0.5 * ( w1[:, :-1, :] * rA[np.newaxis, :-1, :] \
                   +w1[:, 1: , :] * rA[np.newaxis, 1: , :] )
    #- advective flux -
    # surface layer 
    ADVrE_Vm = np.zeros([nr, ny, nx])
    ADVrE_Vm[0, :, :] = rTransV[0, :, :] * v2[0, :, :]
    #ADVrE_Vm[0, :, :] = 0.0         # rigid lid, for checking
    # interior flux
    ADVrE_Vm[1:, :, :] = rTransV[1:, :, :] * \
            0.5 * ( v2[1:, :, :] + v2[:-1, :, :])
    # (linear) Free-surface correction at k>1
    ADVrE_Vm[1:, 1:, :] = ADVrE_Vm[1:, 1:, :] \
            + 0.25 * (\
              w1[1:, 1:, :] * rA[np.newaxis, 1:, :] *\
                (maskC[1:, 1:, :] - maskC[:-1, 1:, :]) \
             +w1[1:, :-1, :] * rA[np.newaxis, :-1, :] *\
                (maskC[1:, :-1, :] - maskC[:-1, :-1, :]) \
                     ) * v2[1:, 1:, :]

    #-- flux divergence (mom_fluxform.F) --
    #- zonal -
    gVx = np.zeros([nr, ny, nx])
    gVx[:, :, :-1] = - 1 / (hS * drF * rAs[np.newaxis, :, :])[:, :, :-1] \
            * (ADVx_Vm[:, :, 1:] - ADVx_Vm[:, :, :-1])
    #- meridional -
    gVy = np.zeros([nr, ny, nx])
    gVy[:, 1:, :] = - 1 / (hS * drF * rAs[np.newaxis, :, :])[:, 1:, :] \
            * (ADVy_Vm[:, 1:, :]-ADVy_Vm[:, :-1, :])
    #- vertical -
    gVz = np.zeros([nr, ny, nx])
    gVz[:-1, :, :] = - 1 / (hS * drF * rAs[np.newaxis, :, :])[:-1, :, :] \
            * (ADVrE_Vm[:-1, :, :]-ADVrE_Vm[1:, :, :])
    gVz[-1, :, :] = - 1 / (hS * drF * rAs[np.newaxis, :, :])[-1, :, :] \
            * (ADVrE_Vm[-1, :, :] - 0.0)     #no bottom flux at bottom cell
    #- total -
    Vm_Advec = gVx + gVy + gVz


    #-- Coriolis and metric term --
    Vm_Cori = np.zeros([nr, ny, nx])
    Vm_metr = np.zeros([nr, ny, nx])
    #- Coriolis (from mom_v_coriolis.F) -
    if cori:
      print("Include Coriolis term in Vm_Advec (following MITgcm diagnostic package convention)")
      print("TO BE DONE (07/21/2023)")
    #- Metric (from mom_v_metric_sphere.F) -
    if metr:
      print("Include Metric term in Vm_Advec (following MITgcm diagnostic package convention)")
      tanPhiAtV = np.zeros([ny, nx])
      tanPhiAtV[:, :-1] = np.tan( np.deg2rad( 0.5*(yG[:, :-1]+yG[:, 1:]) ) )
      #
      Vm_metr[:, 1:, :-1] = -recip_rSphere              \
          *( 0.25*( u1[:, :-1, :-1] + u1[:, :-1, 1:]  \
                   +u1[:, 1: , :-1] + u1[:, 1: , 1:]) \
           )**2                                         \
          *tanPhiAtV[1:, :-1]

    return Vm_Advec, ADVx_Vm, ADVy_Vm, ADVrE_Vm, Vm_Cori, Vm_metr
