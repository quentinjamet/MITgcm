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


def advec_um(uuu, vvv, www, grd, cori=False, metr=False):

    '''
    Compute the 3D advection and advective fluxes
    of zonal momentum.

    Input:
       - u, v, w: three dimensional velocity field
       - grd: list of model grid varaibles
    Output:
       - ADVx_Um, ADVy_Um, ADVrE_Um: zonal, meridional and vertical u-momentum flux
       - Um_Advec: u-momentum flux divergence.
                   In MITgcm diagnostics, Um_Advec also includes 
                   Coriolis (Um_Cori) and metric term (Um_metr).
                   They can be computed and added with cori=True and metr=True (to be done)
    '''

    #-- grid --
    dxG = grd["DXG"]
    dyG = grd["DYG"]
    drF = grd["DRF"]
    hW  = grd["hFacW"]
    hS  = grd["hFacS"]
    hC  = grd["hFacC"]
    rA  = grd["RAC"]
    rAs = grd["RAW"]
    #
    xA = dyG[np.newaxis, :, :] * drF * hW
    yA = dxG[np.newaxis, :, :] * drF * hS
    maskC = hC * 1.0
    maskC[ np.where(maskC>0 ) ] = 1.0

    #-- get and check dimensions --
    [nr, ny, nx] = hS.shape
    tmp = nr*ny*nx
    if np.size(uuu)!=np.size(vvv) or np.size(uuu)!=np.size(www) or np.size(uuu)!=tmp:
      raise ValueError("advec_vm: velocity field do not have the same/right dimension")
    #

    #-- zonal advective flux of U --
    #- transport -
    uTrans = tmpu * xA
    #- adv flux -
    AdvectFluxUU = np.zeros([nr, ny, nx])
    AdvectFluxUU[:, :, :-1] = \
            0.25 *(uTrans[:, :, :-1] + uTrans[:, :, 1:] ) \
                 *(  tmpu[:, :, :-1] +   tmpu[:, :, 1:]   )
    
    #-- meridional advective flux of U --
    #- transport -
    vTrans = tmpv * yA
    #- adv flux -
    AdvectFluxVV = np.zeros([nr, ny, nx])
    AdvectFluxVV[:, 1:, 1:] = \
            0.25 *(vTrans[:, 1:, 1:] + vTrans[:, 1:, :-1] ) \
                 *(  tmpu[:, 1:, 1:] +   tmpu[:, :-1, 1:]   )
    
    #-- vertical advective flux of U --
    #- transport -
    # rTransU :: vertical transport (above U point) 
    rTransU = np.zeros([nr, ny, nx])
    rTransU[:, :, 1:] = \
            0.5 * ( tmpw[:, :, :-1] * rA[np.newaxis, :, :-1] \
                   +tmpw[:, :, 1: ] * rA[np.newaxis, :, 1: ] )
    #- advective flux -
    # surface layer 
    advectiveFluxWU = np.zeros([nr, ny, nx])
    advectiveFluxWU[0, :, :] = rTransU[0, :, :] * tmpu[0, :, :]
    #advectiveFluxWV[0, :, :] = 0.0         # rigid lid, for checking
    # interior flux
    advectiveFluxWU[1:, :, :] = rTransU[1:, :, :] * \
            0.5 * ( tmpu[1:, :, :] + tmpu[:-1, :, :])
    # (linear) Free-surface correction at k>1
    advectiveFluxWU[1:, :, 1:] = advectiveFluxWU[1:, :, 1:] \
            + 0.25 * (\
              tmpw[1:, :, 1:] * rA[np.newaxis, :, 1:] *\
                (maskC[1:, :, 1:] - maskC[:-1, :, 1:]) \
             +tmpw[1:, :, :-1] * rA[np.newaxis, :, :-1] *\
                (maskC[1:, :, :-1] - maskC[:-1, :, :-1]) \
                     ) * tmpu[1:, :, 1:]
        
    
    #-- flux divergence --
    #- zonal -
    gUx = np.zeros([nr, ny, nx])
    gUx[:, :-1, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:, :-1, :] \
            * (ADVx_Um[:, 1:, :] - ADVx_Um[:, :-1, :])
    #- meridional -
    gUy = np.zeros([nr, ny, nx])
    gUy[:, :, 1:] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:, :, 1:] \
            * (ADVy_Um[:, :, 1:]-ADVy_Um[:, :, :-1])
    #- vertical -
    gUz = np.zeros([nr, ny, nx])
    gUz[:-1, :, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[:-1, :, :] \
            * (ADVrE_Um[:-1, :, :]-ADVrE_Um[1:, :, :])
    gUz[-1, :, :] = - 1 / (hW * drF * rAw[np.newaxis, :, :])[-1, :, :] \
            * (ADVrE_Um[-1, :, :] - 0.0)     #no bottom flux at bottom cell
    if cori:
      print("Include Coriolis and metric terms in Um_Advec (following MITgcm diagnostic package convention)")
      print("TO BE DONE (07/21/2023)")
    #- total -
    Um_Advec = gUx + gUy + gUz

    return Um_Advec, ADVx_Um, ADVy_Um, ADVrE_Um


def advec_vm(uuu, vvv, www, grd, cori=False, metr=False):

    '''
    Compute the 3D advection and advective fluxes
    of meridional momentum.

    Input:
       - u, v, w: three dimensional velocity field
       - grd: list of model grid varaibles
    Output:
       - ADVx_Vm, ADVy_Vm, ADVrE_Vm: zonal, meridional and vertical v-momentum flux
       - Vm_Advec: v-momentum flux divergence.
                   In MITgcm diagnostics, Vm_Advec also includes 
                   Coriolis (Vm_Cori) and metric term (Vm_metr).
                   They can be computed and added with cori=True and metr=True (to be done)
    '''

    #-- grid --
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

    #-- get and check dimensions --
    [nr, ny, nx] = hS.shape
    tmp = nr*ny*nx
    if np.size(uuu)!=np.size(vvv) or np.size(uuu)!=np.size(www) or np.size(uuu)!=tmp:
      raise ValueError("advec_vm: velocity field do not have the same/right dimension")
    #
    
    #-- zonal advective flux of V --
    #- transport -
    uTrans = uuu * xA
    #- adv flux -
    ADVx_Vm = np.zeros([nr, ny, nx])
    ADVx_Vm[:, 1:, 1:] = \
        0.25 *(uTrans[:, 1:, 1:] + uTrans[:, :-1, 1:] ) \
             *(   vvv[:, 1:, 1:] + vvv[:, 1:, :-1]    )

    #-- meridional advective flux of V --
    #- transport -
    vTrans = vvv * yA
    #- adv flux -
    ADVy_Vm = np.zeros([nr, ny, nx])
    ADVy_Vm[:, :-1, :] = \
            0.25 *(vTrans[:, :-1, :] + vTrans[:, 1:, :] ) \
                 *(   vvv[:, :-1, :] + vvv[:, 1:, :]   )

    #-- vertical advective flux of V --
    #- transport -
    # rTransV :: vertical transport (above V point) 
    rTransV = np.zeros([nr, ny, nx])
    rTransV[:, 1:, :] = \
            0.5 * ( www[:, :-1, :] * rA[np.newaxis, :-1, :] \
                   +www[:, 1: , :] * rA[np.newaxis, 1: , :] )
    #- advective flux -
    # surface layer 
    ADVrE_Vm = np.zeros([nr, ny, nx])
    ADVrE_Vm[0, :, :] = rTransV[0, :, :] * vvv[0, :, :]
    #ADVrE_Vm[0, :, :] = 0.0         # rigid lid, for checking
    # interior flux
    ADVrE_Vm[1:, :, :] = rTransV[1:, :, :] * \
            0.5 * ( vvv[1:, :, :] + vvv[:-1, :, :])
    # (linear) Free-surface correction at k>1
    ADVrE_Vm[1:, 1:, :] = ADVrE_Vm[1:, 1:, :] \
            + 0.25 * (\
              www[1:, 1:, :] * rA[np.newaxis, 1:, :] *\
                (maskC[1:, 1:, :] - maskC[:-1, 1:, :]) \
             +www[1:, :-1, :] * rA[np.newaxis, :-1, :] *\
                (maskC[1:, :-1, :] - maskC[:-1, :-1, :]) \
                     ) * vvv[1:, 1:, :]

    #-- flux divergence --
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
    if cori:
      print("Include Coriolis and metric terms in Vm_Advec (following MITgcm diagnostic package convention)")
      print("TO BE DONE (07/21/2023)")
    #- total -
    Vm_Advec = gVx + gVy + gVz

    return Vm_Advec, ADVx_Vm, ADVy_Vm, ADVrE_Vm


