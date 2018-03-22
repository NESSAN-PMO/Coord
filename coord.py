#
# --*-- encoding: utf-8 --*--

import numpy as np
import astropy
import astropy.units as u
from astropy.coordinates.earth_orientation import rotation_matrix
from astropy.time import Time
from astropy import _erfa as erfa

iers = astropy.utils.iers.IERS_A.open('finals2000A.all')

def mat_icrs_itrs_equ(epoch, full=False):
    """
        Equinox-based ICRS to ITRS transformation. Employ the IAU 1976 
    precession and IAU 1980 nutation theory.

    Parameter:
    ==========
    epoch: `Time`, scale='utc'
    full: `bool`, default=False
        If full=True, then return approximate rate of mat

    Return:
    =======
    mat: mat = mat_pm * mat_er * mat_nut * mat_pre
    mat_rate: if full=True, return this
        mat_rate = mat_pm * d(mat_er)/dt * mat_nut * mat_pre

    Remark:
    =======
    r_itrs = mat * r_icrs, where vectors are of shape (3,1)
    v_itrs = (mat + mat_rate) * v_icrs
    """
    dut1 = iers.ut1_utc(epoch.jd1, epoch.jd2)
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1)
    tai1, tai2 = erfa.utctai(epoch.jd1, epoch.jd2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    mat_nut = np.matrix(erfa.nutm80(tt1, tt2))
    mat_pre = np.matrix(erfa.pmat76(tt1, tt2))
    s = erfa.sp00(tt1, tt2)
    x, y = iers.pm_xy(epoch.jd1, epoch.jd2)
    mat_pm = np.matrix(erfa.pom00(x.to_value('rad'), y.to_value('rad'), s))
    era = erfa.gst94(ut11, ut12)
    mat_er = rotation_matrix(era*u.rad, 'z')

    mat = mat_pm * mat_er * mat_nut * mat_pre
    if not full:
        return mat
    else:
        mat_tmp = np.matrix('0, 1, 0; -1, 0, 0; 0, 0, 0', dtype=float) 
        mat_er_rate = 7.292115e-5 * mat_tmp * mat_er  
        mat_rate = mat_pm * mat_er_rate * mat_nut * mat_pre
        return mat, mat_rate
    

def mat_teme_tod(epoch):
    """
        TEME to TOD transformation

    Parameter:
    ==========
    epoch: `Time`, scale='utc'

    Return:
    =======
    mat: Rz(-eqeq94)

    """
    tai1, tai2 = erfa.utctai(epoch.jd1, epoch.jd2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    tdb1, tdb2 = tt1, tt2 # time difference between tt and tdb is as much as about 1.6 millisends and are periodic with an average of zero
    theta = erfa.eqeq94(tdb1, tdb2)
    mat = rotation_matrix(-theta*u.rad, 'z')
    return np.matrix(mat)

def mat_icrs_tod(epoch):
    """
        ICRS to TOD transformation

    """
    dut1 = iers.ut1_utc(epoch.jd1, epoch.jd2)
    ut11, ut12 = erfa.utcut1(epoch.jd1, epoch.jd2, dut1)
    tai1, tai2 = erfa.utctai(epoch.jd1, epoch.jd2)
    tt1, tt2 = erfa.taitt(tai1, tai2)
    mat_nut = np.matrix(erfa.nutm80(tt1, tt2))
    mat_pre = np.matrix(erfa.pmat76(tt1, tt2))
    mat = mat_nut * mat_pre
    return mat


def icrs2itrs(epoch, r_icrs, v_icrs=None):
    """
    Transform pos and vel (optional) from icrs to itrs, equinox-based.

    Parameter:
    ==========
    epoch: `Time`, scale='utc'
    pos: `numpy.array`, shape=(3,) 
    vel: `numpy.array`, shape=(3,)

    Return:
    =======
    r_itrs:
    v_itrs:
    """
    if v_icrs is None:
        mat = mat_icrs_itrs_equ(epoch)
        r_itrs = mat * np.matrix(r_icrs).transpose()
        return r_itrs.getA1()
    else:
        mat, mat_rate = mat_icrs_itrs_equ(epoch, True)
        r_itrs = mat * np.matrix(r_icrs).transpose()
        v_itrs = mat * np.matrix(v_icrs).transpose() \
                 + mat_rate * np.matrix(r_icrs).transpose()
        return r_itrs.getA1(), v_itrs.getA1()

def itrs2icrs(epoch, r_itrs, v_itrs=None):
    if v_itrs is None:
        mat = mat_icrs_itrs_equ(epoch)
        r_icrs = mat.transpose() * np.matrix(r_itrs).transpose()
        return r_icrs.getA1()
    else:
        mat, mat_rate = mat_icrs_itrs_equ(epoch, True)
        r_icrs = mat.transpose() * np.matrix(r_itrs).transpose()
        v_icrs = mat.transpose() * np.matrix(v_itrs).transpose() \
                 + mat_rate.transpose() * np.matrix(r_itrs).transpose()
        return r_icrs.getA1(), v_icrs.getA1()


def teme2tod(epoch, r_teme):
    mat = mat_teme_tod(epoch)
    #mat = mat_teme_tod
    r_tod = mat * np.matrix(r_teme).transpose()
    return r_tod.getA1()


def tod2teme(epoch, r_tod):
    mat = mat_teme_tod(epoch)
    r_teme = mat.transpose() * np.matrix(r_tod).transpose()
    return r_teme.getA1()


def icrs2tod(epoch, r_icrs):
    mat = mat_icrs_tod(epoch)
    r_tod = mat * np.matrix(r_icrs).transpose()
    return r_tod.getA1()


def tod2icrs(epoch, r_tod):
    mat = mat_icrs_tod(epoch)
    r_icrs = mat.transpose() * np.matrix(r_tod).transpose()
    return t_icrs.getA1()


def site_location(elong, phi, height):
    """
    Return site position vector in cartesian coordintate in ITRS system,
    based on WGS84 reference ellipsoid

    Parameter:
    ==========
    elong, phi, height: double arrays of sites' geodetic cooradinates

    Return:
    =======
    location: double array

    """
    n = 1  # WGS84
    # n = 2 GRS80
    # n = 3 # WGS72
    return erfa.gd2gc(n, elong, phi, height)


def site_posvel(epoch, elong, phi, height):
    """
    Return site posvel vector in GCRS cartesian system, based on WGS84
    system

    Parameter:
    ==========
    epoch: `Time`, scale='utc'
    elong, phi, height: double arrays of sites' geodetic cooradinates
    
    Return:
    =======
    pos, vel: position and velocity vectors

    """
    location = site_location(elong, phi, height)
    velocity = np.zeros(location.shape)
    pos, vel = itrs2icrs(epoch, location, velocity)
    return pos, vel



if __name__ == '__main__':
    from datetime import datetime
    # r_itrs = np.array([19440.953805, 16881.609273, -6777.115092])
    # v_itrs = np.array([-0.8111827456, -0.2573799137, -3.0689508125])
    epoch = Time(datetime(2017,12,17,12, 3, 38, 265000))
    # mat_teme_tod = mat_teme_tod(epoch)
    # mat_icrs_tod = mat_icrs_tod(epoch)
    # print(mat_teme_tod)
    # print(mat_icrs_tod)
    # print(mat_icrs_tod.transpose()*mat_teme_tod)
#    r_icrs, v_icrs = itrs2icrs(epoch, r_itrs, v_itrs)
    r_teme = np.array([1.0, 0.0, 0.0])
    r_tod = teme2tod(epoch, r_teme)
    #r_icrs = tod2icrs(epoch, r_tod)
    print(r_tod)

