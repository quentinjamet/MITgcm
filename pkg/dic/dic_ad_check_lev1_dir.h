C $Header: /u/gcmpack/MITgcm/pkg/dic/dic_ad_check_lev1_dir.h,v 1.1 2008/04/24 21:32:13 gforget Exp $
C $Name:  $

#ifdef ALLOW_DIC

C common CARBON_NEEDS
CADJ STORE pH                = comlev1, key = ikey_dynamics
CADJ STORE fice              = comlev1, key = ikey_dynamics
CADJ STORE silica            = comlev1, key = ikey_dynamics
CADJ STORE atmospco2         = comlev1, key = ikey_dynamics

C common dic_load
CADJ STORE dicwind0,dicwind1   = comlev1, key = ikey_dynamics
CADJ STORE ice0,ice1         = comlev1, key = ikey_dynamics
CADJ STORE atmosp0,atmosp1   = comlev1, key = ikey_dynamics
CADJ STORE silica0,silica1   = comlev1, key = ikey_dynamics
#ifdef ALLOW_FE
CADJ STORE feinput0,feinput1 = comlev1, key = ikey_dynamics
#endif

CADJ STORE gsm_s             = comlev1, key = ikey_dynamics
CADJ STORE co2atmos          = comlev1, key = ikey_dynamics
CADJ STORE total_atmos_carbon  = comlev1, key = ikey_dynamics

#endif /* ALLOW_DIC */

