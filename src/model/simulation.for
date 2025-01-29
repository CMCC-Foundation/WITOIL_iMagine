c-----------------------------------------------------------------------------------
c  MEDSLIK-II_3.00
c  oil spill fate and transport model 
c-----------------------------------------------------------------------------------
c  medslik_II.for
c  This program simulates the tranport and weathering of an oil spill.
c-----------------------------------------------------------------------------------
c  Copyright (C) <2012>
c  This program was originally written
c  by Robin Lardner and George Zodiatis.
c  Then, it was modified by Michela De Dominicis.
c  In the current version,
c  subsequent additions and modifications
c  have been made by Antonio Augusto Sepp Neves (CMCC),
c  Igor Atake (CMCC), Marco Seracini (UniBo) and Francesco M. Benfenati (UniBo).
c----------------------------------------------------------------------------------
c  The development of the MEDSLIK-II model is supported by a formal agreement
c  Memorandum of Agreement for the Operation and Continued Development of MEDSLIK-II
c  signed by the following institutions:
c  INGV - Istituto Nazionale di Geofisica e Vulcanologia
c  OC-UCY - Oceanography Center at the University of Cyprus
c  CNR-IAMC - Consiglio Nazionale delle Ricerche – Istituto per 
c  lo Studio dell’Ambiente Marino Costiero
c  CMCC - Centro Euro-Mediterraneo sui Cambiamenti Climatici
c 
c  This program is free software: you can redistribute it and/or modify
c  it under the terms of the GNU General Public License as published by
c  the Free Software Foundation, either version 3 of the License, or
c  any later version.
c
c  This program is distributed in the hope that it will be useful,
c  but WITHOUT ANY WARRANTY; without even the implied warranty of
c  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
c  GNU General Public License for more details.
c  You should have received a copy of the GNU General Public License
c  along with this program.  If not, see <http://www.gnu.org/licenses/>.
c-----------------------------------------------------------------------------------
      program simulation
      implicit real*8(a-h,o-z)
      character dummy*80

c----------------------------------------------------------------------------
c  DEFITION OF BASIC NETCDF PARAMETERS
c----------------------------------------------------------------------------
      include 'netcdf.inc'
c     This is the name of the data file we will create.
      character*(*) FILE_NAME
      parameter (FILE_NAME = 'spill_properties.nc')
      integer ncid
c     Setting up dimensions --- n_parcel and time
      integer NDIMS
      parameter (NDIMS = 2)
      character*(*) PARCEL_NAME, TIME_NAME
      parameter (PARCEL_NAME = 'parcel_id')
      parameter (TIME_NAME = 'time')
c     Error handling.
      integer nc_status  
c----------------------------------------------------------------------
c     variables used for netcdf files size calculation
c---------------------------------------------------------------------- 
      character infile*400
      character experiment_folder*400,preproc_folder*400
      integer len_dir,dimid,imx_curr,jmx_curr,
     &      mm,nm,ncid_nf,imx_bath,jmx_bath
      character*(*) LAT_DIM_NAME, LON_DIM_NAME
      parameter (LON_DIM_NAME='nav_lon', LAT_DIM_NAME='nav_lat')
c----------------------------------------------------------------------
c     variables used to allow nm,mm to be passed as arguments
c---------------------------------------------------------------------- 
c     Set the common block
      common /ncfile/ ncid,nc_status
      common /size/ mm,nm
      common /saved/ isub
c     Get argument passed to main
      call getarg(1,experiment_folder)
      call getarg(2,preproc_folder)
c     Create the output file. 
      nc_status = nf_create(
     &  TRIM(experiment_folder)//"/out_files/"//FILE_NAME,
     &  NF_NETCDF4, ncid)
      if (nc_status .ne. nf_noerr) call handle_err(nc_status)
c----------------------------------------------------------------------
c  NETCDF FILE HAS BEEN CREATED
c-----------------------------------------------------------------------

c--------------------------------------------------------------------
c     Calculate the values of mm and nm for the subroutines
c--------------------------------------------------------------------
c set the netcdf file to read (one in the available set)
      infile=TRIM(preproc_folder)//'/bnc_files/dtm.bath' 
c open bathymetry file
      open(unit=999,file=infile,status='old',action='read')
      read(999,'(a80)') dummy
      read(999,*) empty,empty,empty,empty
      read(999,*) imx_bath,jmx_bath
      close(999)
      print *, "Ocean grid has shape (lon, lat) ", imx_bath, jmx_bath
      mm = imx_bath
      nm = jmx_bath 
      call main(experiment_folder,preproc_folder)
      end program simulation

c----------------------------------------------------------------------
c     begin the main program
c----------------------------------------------------------------------
      subroutine main(experiment_folder,preproc_folder)
      implicit real*8(a-h,o-z)
      parameter(ntm=2000,npc=100000,nss=200000,msp=3000)
      integer mm,nm
      common /size/ mm,nm
c----------------------------------------------------------------------
c     dimension declarations for environment
c----------------------------------------------------------------------

      real*8,dimension(mm,nm) :: us,vs,dus,dvs,   
     &      sst,wx,wy,
     &      u10,v10,u30,v30,
     &      u120,v120,dsst,dwx,dwy,
     &      du10,dv10,
     &      du30,dv30,du120,dv120,
     &      dwinx,dwiny 
      dimension 
     &      h(mm,nm),itype(mm,nm),uadv(mm,nm),vadv(mm,nm)
      dimension 
     &      wndtim(0:ntm),wndvel(0:ntm),wnddir(0:ntm),
     &      wcx(0:11,50),wcy(0:11,50),wcu(0:11,50),wcv(0:11,50),
     &      winx(mm,nm),winy(mm,nm),wdrftx(mm,nm),wdrfty(mm,nm),
     &      stoku(mm,nm),stokv(mm,nm),wavenum(mm,nm),
     &      wvel_vec(720),wvel_mean(720),wdir_vec(720),wdir_mean(720),
     &      freq_sp(700),fang(700),erre(700),sp_exp1(700), spectra(700),
     &      wave_num(700),stoke_sp(700),hwave_d(700),stoke_d(700),
     &      sp_exp2(700)
      dimension 
     &      nwc(0:10),uu(24),vv(24),ss(3),
     &      curtim(0:ntm),curvel(0:ntm),curdir(0:ntm),
     &      crntim(5),crnu(5),crnv(5),
     &      ifcstfn(720),fcsttim(720),iwfcstfn(30),wfcsttim(30,24)  
      dimension 
     &      bmtim(20),bmlat1(20),bmlon1(20),bmlat2(20),bmlon2(20),
     &      bmx1(20),bmy1(20),bmx2(20),bmy2(20),ibmeff(20)
c----------------------------------------------------------------------
c     dimension declarations for oil slick
c----------------------------------------------------------------------
      real,dimension(nss) :: seep
      integer bsegxxyy1_id, bsegxxyy2_id
      integer, dimension(npc) :: is
      integer, dimension(npc) :: bseg1_id,bseg2_id,
     &      itmp_bseg1_id,itmp_bseg2_id
      integer, dimension(nss) :: bseg1,bseg2,
     &       bseg1_pid,bseg2_pid,ibd 
      dimension 
     &      blat_filt(10000),blon_filt(10000),
     &      c1p(npc),c2p(npc),px(npc),py(npc),pz(npc),
     &      px_contour(npc), py_contour(npc),
     &      ib(npc),itmp(npc),bd1(nss,4),
     &      seg(nss,4),alngt(nss),segx(nss),segy(nss),ns0(nss),
     &      prel(nss),sfrac(nss),vcst(nss),
     &      den(msp),vis(msp),visem(msp),tre(msp),c1ms(msp),c2ms(msp),
     &      atn(msp),atk(msp),ato(msp),ttk(msp),tto(msp),
     &      vtn(msp),vtk(msp),vto(msp),xcl(msp),xcs(msp),
     &      vtne(msp),vtke(msp),vte(msp),vtnd(msp),vtkd(msp),vtd(msp),
     &      ftk(msp),ftn(msp),fw(msp),pcte(msp),pctd(msp)
c----------------------------------------------------------------------
c     common declarations
c----------------------------------------------------------------------
      common /blk3/ delx,dely,pi,degrad
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      common /blk5/ mzb,mzf,nzb,nzf
c----------------------------------------------------------------------
c     common declarations for oil & spill characteristics
c----------------------------------------------------------------------
      common /spill/  idd,imm,iyr,ispill,tstart,tcomp,x0,y0
      common /evap/ ce1,ce,vappr,fmaxe
      common /disp/ cd1,cd3,cd4,cd5,vl,vs1,um,stk,stn,fmaxd
      common /emul/ cm1,cm2,cm3
      common /sprd/ cs1,cs2,cs3
      common /phys/ deno,denk,cdt,viso,visk,cvt,tvk0,denw,tk,tk0,wvel
      common /weath/ ned
c---------------------------------------------------------------------
c     common declarations of the netcdf file
c---------------------------------------------------------------------
      common /ncfile/ ncid,nc_status
c----------------------------------------------------------------------
c     character declarations
c---------------------------------------------------------------------- 
      character empty*80, a1*1, a2*2, a3*3, a4*4, 
     &          ay*2, am*2, ad*2, ah*2, d24*3, d06*3, vlunit*4, pref*3,
     &          nore*2, nora*2,ora*2,ore*2,dummy*80 
      character fn(3)*6,fnsim(30)*17, fnadm(30)*22, fncym(30)*19
      character seas1*1, seas2*1, fcstcurdir*400,list*6
      character fcstfn(720)*16, wfcstfn(30)*6, date*10
      character experiment_folder*400,preproc_folder*400
      integer nt, ix, count, id_p, pxi, pyi
      logical ex, last_release
c----------------------------------------------------------------------
c     DECLARING NETCDF VARIABLE AND DIMENSION IDS
c----------------------------------------------------------------------
c     We recommend that each variable carries a "units" attribute.
c     Oil density and number of parcels were also added for 
c     further calculations
      character*(*) UNITS,OIL_DENSITY,PARCEL_VOL
      character*(*) SP_LON,SP_LAT
      parameter (UNITS = 'units')
      parameter (OIL_DENSITY = 'oil_density')
      parameter (PARCEL_VOL = 'volume_of_parcel')
      parameter (SP_LON = 'initial_position_x')
      parameter (SP_LAT = 'initial_position_y')
      character*(*) TIME_UNITS, EVOL_UNITS, LAT_UNITS, LON_UNITS
      character*(*) NVOL_UNITS, PRTS_UNITS, WC_UNITS,TFXD_UNITS
      character*(*) VEM_UNITS,VIS_UNITS,DEN_UNITS,VOLR_UNITS  
      character*(*) BV_UNITS,BSEG1_UNITS,BSEG2_UNITS
      parameter (BV_UNITS = 'm3')
      parameter (TIME_UNITS = 'hours')
      parameter (EVOL_UNITS = 'm3')
      parameter (NVOL_UNITS = 'm3')             ! check that 
      parameter (PRTS_UNITS = '0,1,2,3,4,-nsg,9')
      parameter (BSEG1_UNITS = 'unique_id_in_coastline_file_extreme1')
      parameter (BSEG2_UNITS = 'unique_id_in_coastline_file_extreme2')
      parameter (LAT_UNITS = 'degrees_north')
      parameter (LON_UNITS = 'degrees_east')
      parameter (WC_UNITS = 'percentage')
      parameter (TFXD_UNITS = 'tonnes')
      parameter (VEM_UNITS = 'Pa.s')
      parameter (VIS_UNITS = 'Pa.s')
      parameter (DEN_UNITS = 'kg/m3')
      parameter (VOLR_UNITS = 'percentage')
c     The start and count arrays will tell the netCDF library where to
c     write our data.  
      integer NPARCEL,NTIME
      integer NDIMS
      parameter (NDIMS = 2)
      integer prtcl_dimid, time_dimid, nc_time
      integer ncpointer(NDIMS),ncounter(NDIMS),dimids(NDIMS)
      real x_coordinate,y_coordinate, ntons
      real*8 iage, idepth, delt
c     declaring variables IDs
      integer bv_varid, evol_varid, nvol_varid, wc_varid,
     &        lat_varid, lon_varid, prtt_varid, tfxd_varid,
     &        fwf_varid, fwi_varid, vemi_varid, time_varid,
     &        vemf_varid, deni_varid, denf_varid, visi_varid,
     &        visf_varid, volr_varid, bseg1_varid, bseg2_varid
      integer iseed
c----------------------------------------------------------------------
c     transformations betweenthe medslik grid coords and lat/lon
c----------------------------------------------------------------------
      xgrid(alon)=(alon-along1)/dlong+1.d0
      ygrid(alat)=(alat-alatg1)/dlatg+1.d0
      glon(x)=along1+(x-1.d0)*dlong
      glat(y)=alatg1+(y-1.d0)*dlatg
      pi=4.*datan(1.d0)
      degrad=180.d0/pi
      open(90,file=TRIM(experiment_folder)//'/medslik.log')
c-----------------------------------------------------------------------
c      model options (M. De Dominicis)
c----------------------------------------------------------------------
c     model parameters        ! default values
c     istoke= 0:no stokes drift calculation; 1: stoke drift calculation using Jonswap spectrum (M. De Dominicis)
c     alpha, beta = empirical parameters for wind drift of slick
c     ibetared = 1 if beta reduces by 50% at wind speed = halfspeed
c     beta=beta0*(1.-ibetared*wvel/(wvel+halfspeed))
c     iwindred = 1 if fraction of forecast wind is subtracted in drift formula
c     wredfrac = fraction of fcst wind to be subtracted when fcst currents are used 
c     ismag = 1 if Smagorinski scheme is used for horiz diffusivity
c     horizk = horizontal diffusivity
c     vertk1,2 = vertical diffusivities above & below thermocline
c     thermocl = depth of thermocline
c     ntot = no of parcels used to model diffusion and dispersion
c     fcstdepth1,2,3 = depths at which currents are printed in forecast files
c----------------------------------------------------------------------
      open(39,file=TRIM(experiment_folder)//
     &             '/xp_files/parameters.txt',status='old')
      read(39,*) empty 
      read(39,*) istoke        ! 01
      print *, "Stokes drift (1=True, 0=False)",istoke
      read(39,*) alpha        ! 0.031
      read(39,*) beta0        ! 13.
      read(39,*) ibetared     ! 00
      read(39,*) halfspeed    ! 0.0
      read(39,*) iwindred     ! 00
      read(39,*) wredfrac     ! 0.0
      read(39,*) ismag        ! 00
      read(39,*) horizk       ! 2.0
      read(39,*) vertk1       ! 0.01
      read(39,*) vertk2       ! 0.0001
      read(39,*) thermocl     ! 30.0
      read(39,*) ntot         ! 25000
      read(39,*) fcstdep1,fcstdep2,fcstdep3     ! 10.,30.,120.
      read(39,*) idepth       ! 00= only superficial
      if(ntot.gt.100000) then
        write(6,*) 'Total number of parcels cannot exceed 100,000'
        write(6,*) 'Return to Parameters
     &         Form and change this number'
        stop
      endif

c----------------------------------------------------------------------
c     evaporation constants from Mackay et al
c     ce=coeff accounts for drop in vapour pressure with evaporation (ce=10-20)
c          ce1=akew*(wvel*3.6)**gamma_evap = evaporative exposure to wind
c     visk = coeff for change of viscosity with evaporation
c----------------------------------------------------------------------
      read(39,*) empty
      read(39,*) ce     ! 12.0
      read(39,*) akew   ! 0.000033
      read(39,*) gamma_evap  ! 0.78
      read(39,*) visk   ! 4.
c----------------------------------------------------------------------
c     emulsion constants from Mackay et al
c     cm1 controls the effect of water fraction on mousse viscosity
c     cm2 controls the rate of water absorption
c     cm3 controls maximum water fraction in the mousse (decreased for heavy oils)
c     icm3 = 1 if max water fraction increases/decreases for light/heavy oils
c----------------------------------------------------------------------
      read(39,*) empty
      read(39,*) cm1     ! 0.65
      read(39,*) cm2     ! 1.6e-06
      read(39,*) cm3     ! 1.333
      read(39,*) icm3    ! 1
c----------------------------------------------------------------------
c     dispersion constants from Mackay et al
c     cd1=downward diffusion velocity of small droplets (m/s)
c     cd3=controls the rate of dispersion of all droplets by waves
c     cd4=controls the fraction of droplets below the critical size
c     cd5=controls the dispersion from the thin slick (sheen)
c     vs1=rising velocity of small droplets (m/s)
c     vl=rising velocity of large droplets (m/s)
c     um=controls depth of well-mixed surface layer (m)
c     st=interfacial tension between oil and water
c     fmaxd=max dispersive fraction
c----------------------------------------------------------------------
      read(39,*) empty
      read(39,*) cd1     ! 0.001
      read(39,*) cd3     ! 0.8e-05
      read(39,*) cd4     ! 50.0
      read(39,*) cd5     ! 2000.0
      read(39,*) vl      ! 0.08
      read(39,*) vs1     ! 0.0003
      read(39,*) um      ! 0.5
      read(39,*) st      ! 24.0
      stk=st
      stn=st
c----------------------------------------------------------------------
c     spreading constants from Mackay et al
c     sprdmx = max time for spreading of spill (hours)
c     for new parameter files read the following:
c     seepmx = limiting concentration on coast (tons/km)
c     apicoeff = coeff of reduction of coastalretention rate for heavy oils
c----------------------------------------------------------------------
      read(39,*) empty
      read(39,*) cs1     ! 1.0
      read(39,*) cs2     ! 150.0
      read(39,*) cs3     ! 0.0015
      sprdmx=24.d0
      data seepmx /5000.d0/
      read(39,*) empty
      read(39,*) seepmx     ! 5000.0
      read(39,*) apicoeff   ! 0.0
c     Computational parameters
      read(39,*) empty
      read(39,*) deltsc ! time step seconds
      close(39)
c----------------------------------------------------------------------
c     delt = time increment for spill re-computations (hrs)
c     stph, nstph = no of steps per hour
c----------------------------------------------------------------------
      delt = deltsc/3600.d0 ! time_step from seconds to hours
      deltmn=delt*60.d0
      stph=1.d0/delt
      nstph=stph+0.001d0
c----------------------------------------------------------------------
c     irestart = 1 if a previous run is to be restarted
c     ihrestart = number of hours of this previous run
c----------------------------------------------------------------------
      open(40,file=TRIM(experiment_folder)//'/xp_files/oilspill.inp'
     &      ,status='old')
      read(40,*) irestart,ihrestart
      if(irestart.eq.0) ihrestart=0
c----------------------------------------------------------------------
c     idd/imm/iyr = date of spill
c     istart = time of day at start of spill
c     tstart = nearest hour to start of spill
c     tspill (ispill) = duration of spill (hours)
c     splrte = spill rate (tons per hour)
c     tcomp (icomp) = duration of computation from start of spill (hrs)
c     lat,alat, lon,alon = geographical location of spill (deg,min)
c     pref = 3-letter prefix for labelling output files
c     iprs = interval for output (hrs)
c     icrn = 1/0 if observation of spill posn is (is not) to be applied
c----------------------------------------------------------------------
      read(40,'(i2)') idd
      read(40,'(i2)') imm
      read(40,'(i4)') iyr
      read(40,'(i4)') istart
      read(40,'(i4)') ispill
      read(40,*) splat
      read(40,*) splon
      read(40,'(i4)') icomp
      read(40,'(i3)') iprs
      read(40,'(i2)') icrn
      tstart=dfloat(istart/100)
      tspill=dfloat(ispill)
      tcomp=dfloat(icomp)
c----------------------------------------------------------------------
c     vlunit = unit of volume (tons, cu.m, tons, gals)
c     splrte = spill rate (no units per hour) or total volume for ispill=0 
c     api = API number of the oil
c     deno = oil density  (g/cm**3)
c     den2 = density of residual part (g/cm**3)
c     respc = residual percentage
c     viso = initial oil viscosity
c     tem0 = reference temperature for viscosity
c     vappr = oil vapour pressure from file
c     max water content reduced/increased for heavy/light oils
c----------------------------------------------------------------------
      read(40,'(f9.2)') splrte
      read(40,'(a4)') vlunit
c---------------------------------------------------------------------------------
c M. De Dominicis
c      isat= 0:point source; 1: areal source of spill (from manual slick contour
c      or from satellite data)
c      iage= age of the slick (in hours, 0, 24, 48)
c---------------------------------------------------------------------------------
      read(40,*) iage 
      read(40,*) isat
c----------------------------------------------------------------------
c     ibooms = 1/0 if booms are (are not) deployed
c----------------------------------------------------------------------
      read(40,'(i2)') ibooms
      read(40,'(i6)') iseed
      read(40,'(a80)') empty
      read(40,'(f5.2)') api
      read(40,'(f5.3)') deno
      read(40,'(f5.3)') den2
      read(40,'(f5.2)') respc
      read(40,'(f5.1)') viso
      read(40,'(f5.1)') tem0
      read(40,'(f5.1)') vappr
      tvk0=tem0+273.d0
      if(icm3.eq.1.and.cm3.eq.1.333d0) then
        cm3=(10.d0/9.d0)*(2.d0-1.d0/(1.d0+4.d0**(1.d0-api/17.d0)))
      endif

c----------------------------------------------------------------------
c     fcstfn() = name of forecast file of currents
c     ifcstfn() = 1/0 if the file is (is not) available
c     wfcstfn() = name of forecast file of winds
c     iwfcstfn() = 1/0 if the wind file is (is not) available
c----------------------------------------------------------------------
      read(40,'(i4)') nfcst
      nwfcst = nfcst
c----------------------------------------------------------------------
c       Hourly forecast files (M. De Dominicis)
c----------------------------------------------------------------------
      do i=1,nfcst
        read(40,'(a6)') fn(i)
        wfcstfn(i) = fn(i)
        ifcstfn(i) = 1
        iwfcstfn(i) = 1
c       date list for hourly currents files 
c       starting from 00:00
        list=fn(i)
        do nt=1,24
          if(nt.lt.10) then
            write(nore,'(i2)') nt
            nora='0'//nore(2:2)
          else 
            write(nora,'(i2)') nt
          endif               
c       date list for hourly currents files 
c       starting from 12:00 (MFS and AFS)
          if(nt.le.12) then
            count=i
            write(ore,'(i2)') nt+12
            ora=ore(1:2)
          endif
          if(nt.gt.12.and.nt.lt.22) then
            count=i+1
            write(ore,'(i2)') nt-12
            ora='0'//ore(2:2)
          endif
          if(nt.ge.22.and.nt.le.24) then
            count=i+1
            write(ore,'(i2)') nt-12
            ora=ore(1:2)
          endif
          fcstfn((i-1)*24+nt)='merc'//list//nora(1:2)//'.mrc'
          ifcstfn((i-1)*24+nt)=ifcstfn(i)
        enddo
c ---------------------------------------------------------------------
c     Daily forecast files
c     assign directory for forecast current data
c----------------------------------------------------------------------
      enddo
      close(40)
      a4 = fcstfn(1)(1:4)
      fcstcurdir=TRIM(preproc_folder)//'/oce_files/' ! str


c----------------------------------------------------------------------
c     Construct basic grid parameters and bathymetry.
c     delx, dely = bathymetry grid spacing in metres
c         (set grid spacing delx, at mid-latitude of region)
c        slick travels at less than 1.5 kts from spill site.
c----------------------------------------------------------------------    
      open(50,file=TRIM(preproc_folder)//'/bnc_files/dtm.bath',
     &        status='old')
      read(50,'(a80)') empty 
      read(50,*) along1,along2,alatg1,alatg2
      read(50,*) mmaxb, nmaxb
      if(mmaxb.ne.mm.or.nmaxb.ne.nm) then
        print *, "ERROR: Bathymetry
     &             has not correct dimensions"
        print *, "Bathymetry has shape ", mmaxb, nmaxb
        print *, "but it is stored in a matrix with shape ", mm, nm
        stop
      endif
      do n=nm,1,-1
        do m=1,mm 
          read(50,'(i4)') ih1
          if(ih1.eq.9999) then
            itype(m,n)=0
            h(m,n)=0
          else
            itype(m,n)=1
            h(m,n)=float(ih1)
          endif
        enddo
      enddo
      dlong = (along2 - along1) / dfloat(mm-1)
      dlatg = (alatg2 - alatg1) / dfloat(nm-1)
      avlat=(alatg1+alatg2)/2.d0
      dely=dlatg*60.d0*1849.d0
      delx=dlong*60.d0*1849.d0*dcos(avlat/degrad)
      close(50)
c----------------------------------------------------------------------
c     vertd1,2 = vertical  diffusion distance during time delt
c     horizd = horizontal diffusion distance during time delt
c     check stability of horizontal diffusion simulation (ismag=0)
c     ismag = 1 if Smagorinski scheme used for horizd (forecast data only)
c----------------------------------------------------------------------
      vertd1=dsqrt(6.d0*vertk1*delt*3600.d0)
      vertd2=dsqrt(6.d0*vertk2*delt*3600.d0)
      if(ismag.eq.0) then
        horizd=dsqrt(6.d0*horizk*delt*3600.d0)
        dtmx1=delx**2/(6.d0*horizk*3600.d0)
        dtmx2=dely**2/(6.d0*horizk*3600.d0)
        if(delt.gt.dtmx1.or.delt.gt.dtmx2) then
          write(6,*)'delt must not exceed ',dtmx1,' or ',dtmx2,' hrs'
          stop
        endif
      endif
c----------------------------------------------------------------------
c     x0,y0 = location of spill in coords of the bathymetry grid
c----------------------------------------------------------------------
      x0=xgrid(splon)
      y0=ygrid(splat)
  
      m0=int(x0)
      n0=int(y0)
      isum=itype(m0,n0)+itype(m0+1,n0)+
     &      itype(m0,n0+1)+itype(m0+1,n0+1)

      if(m0.lt.1.or.m0.gt.mm-1.or.n0.lt.1.or.n0.gt.nm-1.or.
     &   isum.eq.0) then
        write(6,*) '************************************************'
        write(6,*) ' Spill location is outside the given water body'
        write(6,*) ' You should continue only if using restart file'
        write(6,*) '************************************************'
      else if(isum.le.2) then
        write(6,*) '************************************************'
        write(6,*) 'WARNING:  Spill location is very close to a coast'
        write(6,*) '   and  may be outside the given water body.'
        write(6,*) 'Check its latitude and longitude and if necessary'
        write(6,*) '      move it further into the water body'
        write(6,*) '************************************************'
      endif

      write(6,*) 'The spill simulated has the following parameters:'
      write(6,600) idd,imm,iyr,istart
      write(6,601) splat,splon,x0,y0,tspill,icomp
      write(6, *) "Time step (seconds) ", delt*3600.d0
      write(6, *) "Number of particles", ntot
      if(ispill.eq.0) write(6,603) splrte,vlunit
      if(ismag.eq.0) write(6,604) horizd
  600 format(' Date:           ',i2,'/',i2,'/',i4,'  hour: ',i4.4)
  601 format(' Location:  lat  ',f8.4,',    lon ',f8.4/
     &       ' Grid coords: x  ',f8.4,',      y ',f8.4/
     &       ' Spill duration: ',f7.0,' hours'/
     &       ' Length of run: ',i4,' hours')
  602 format(' Spill rate:    ',f9.2,' ',a4,'/hr')    
  603 format(' Total volume:  ',f9.2,' ',a4)    
  604 format(' Horiz diffusion distance:  ',f12.5, ' meters')  
  770 format(/'Spill positions will be corrected from observations')
      write(90,'(/'' Welcome to the MEDSLIK run module''/)')
      write(90,*) 'The spill simulated has the following parameters:'
      write(90,600) idd,imm,iyr,istart
      write(90,601) splat,splon,x0,y0,tspill,icomp
      write(90, *) "Time step (seconds) ", delt*3600.d0
      write(90, *) "Number of particles", ntot
      if(ispill.gt.0) write(90,602) splrte,vlunit
      if(ispill.eq.0) write(90,603) splrte,vlunit
      if(ismag.eq.0) write(90,604) horizd
      if(icrn.eq.1) then
        write(90,770)
      endif
c----------------------------------------------------------------------
c     ix=random 6-digit seed for random number generator
c----------------------------------------------------------------------
      if(iseed.eq.0) then
        call seedmedslik(ix) ! seed is computed
      else
        ix = iseed ! fixed seed set by the user
      endif
      print *, "Seed used for random numbers generation is", ix
c---------------------------------------------------------------------
c       Initial slick aging - contour slick
c--------------------------------------------------------------------
      !no of timesteps for aging the oil properties
      if(isat.eq.1) then
        nst0=iage/delt
      endif
c----------------------------------------------------------------------
c     totvol = total spilled volume in cube meters
c     nmini = number of incremental mini-spills one mini-spill per delt)
c     vmspl=volume per mini-spill (in cu m)
c
c     ntot = total no of lagrangian parcels spilt (initialy from par-file)
c     vpp = cube meters per lagrangian parcel
c     nppms = no of parcels per mini-spill
c----------------------------------------------------------------------
      if(ispill.gt.0) then 
        totvol=splrte*tspill/deno
      elseif(ispill.eq.0) then 
        totvol=splrte/deno
      else
        stop "Spill duration is negative"
      endif    
      nmini=int(tspill/delt+0.0001d0)+1
      if(tspill.gt.(nmini-1)*delt) nmini=nmini+1
c     Check if nmini exceeds number of minispills allowed
      if(nmini.gt.3000) then
        print *, "ERROR: Number of minispill(s) ", nmini, 
     &           " exceeds the number allowed ", 3000
        print *, "Spill duration is ", tspill, "hours"
        print *, "Every our, ", nstph, 
     &           " minispills are computed (given the time step of ",
     &           delt, "hours"
        print *, "Please, modify *spill duration* or *time step* ", 
     &           "considering the maximum number of minispills allowed."
        print *, "STOP"
        stop
      endif
      ! Minispill and particles volume
      vmspl=totvol/nmini
      nppms=int(dfloat(ntot)/dfloat(nmini)+0.01d0)
      ntot=nppms*nmini
      vpp=totvol/ntot
      write(90,*) 'Total volume (m^3) released = ',totvol
      write(90,*) 'Total no of parcels = ',ntot
      write(90,*) 'Volume (m^3) per parcel     = ',vpp
      write(90,*) 'No of minispills    = ',nmini
      write(90,*) 'Parcels per m_spill = ',nppms
c----------------------------------------------------------------------
c     c1i = fraction of evaporative part
c     c2i = fraction of residual part,   c1i + c2i = 1
c     den1 = density of evaporative part (kg/m**3)
c     den2 = density of residual part (kg/m**3)
c     deno = oil density (kg/m**3):    deno = c1i*den1 + c2i*den2
c     tsat = saturation temp for evaporative component
c     fmaxe=max evaporative fraction
c
c     denk = coeff for change of density with evaporation
c     cdt, cvt = coeffs for change of density & viscosity with temperature
c     denw0 = sea water density at temp tk0 (kg/m3)
c     tk0 = reference temperature (deg Kelvin)
c----------------------------------------------------------------------
      deno=deno*1000.d0
      den2=den2*1000.d0
      c2i=respc/100.d0
      c1i=1.d0-c2i
      den1=(deno-c2i*den2)/c1i
      tsat=400.d0-3.d0*api
      fmaxe=c1i
      fmaxd=c2i
      denk=(den2-den1)/deno   !0.18
      cdt=0.008d0
      cvt=5000.d0
      denw0=1026.d0
      denw=1026.d0
      row=(denw0-deno)/deno
      tk0=293.d0
c----------------------------------------------------------------------
c     maxst = no of steps of computation of length delt each
c     ned = no of calls to evap/disp subroutine per delt
c     tedsec = time interval in secs between evap/disp calls
c     tiprs = time interval (hours) for printing output
c     nprs = step number at which output printing begins
c     iprs = no of steps between output printing
c----------------------------------------------------------------------
      maxst = tcomp*stph+0.001d0
      if(irestart.eq.1) maxst = maxst - ihrestart*stph+0.001d0
      ned=30
      tedsec=deltsc/ned
      tiprs=dfloat(iprs)
      nprs=tiprs*stph+0.001d0
      iprs=tiprs*stph+0.001d0
c----------------------------------------------------------------------
c     initialise wind velocity and wind forecast velocity
c----------------------------------------------------------------------
      do m=1,mm
        do n=1,nm
          wx(m,n)=0.d0
          wy(m,n)=0.d0
          winx(m,n)=0.d0
          winy(m,n)=0.d0
        enddo
      enddo               
c----------------------------------------------------------------------
c     if booms deployed, read boom data
c     bmtim(k) = hour of kth boom deployment relative to 0 hrs on nday 0
c     bmx1(k),bmy1(k),bmx2(k),bmy2(k) = 5 km grid coords of boom ends 
c     ibmeff(k) = efficiency (%) of kth boom 
c     NOTE: booms deployment has not been used and tested 
c     in the MEDSLIK-II_1.0 and following versions 
c----------------------------------------------------------------------
      nbooms=0
      if(ibooms.eq.1) then
        open(48,file=TRIM(experiment_folder)//'/xp_files/medslik.bms',
     &          status='old')
        read(48,'(a80)') empty
        read(48,'(i2)') nbooms
        read(48,'(a80)') empty
        do k=1,nbooms
          read(48,'(i2,1x,i2,1x,i4,2x,i2,3x,i5,4(3x,i2,2x,f5.2),2x,i3)') 
     &            id,im,iy,ihb,lenbm,latdg1,alatm1,londg1,alonm1,
     &            latdg2,alatm2,londg2,alonm2,ibmeff(k)        
          nday = jdiff(idd,imm,iyr,id,im,iy)
          bmtim(k)=nday*24.+ihb
          bmlat1(k)=dfloat(latdg1)+alatm1/60.d0
          bmlon1(k)=dfloat(londg1)+alonm1/60.d0
          bmlat2(k)=dfloat(latdg2)+alatm2/60.d0
          bmlon2(k)=dfloat(londg2)+alonm2/60.d0
          bmx1(k)=xgrid(bmlon1(k))
          bmy1(k)=ygrid(bmlat1(k))
          bmx2(k)=xgrid(bmlon2(k))
          bmy2(k)=ygrid(bmlat2(k))
        enddo    
        close(48)
      endif
c----------------------------------------------------------------------
c     If icrn = 1, read spill correction data and construct correction 
c     velocities in m/s to shift centre of slick from the previous
c     computed position (xold,yold) to the observed position (xnew,ynew)
c     For 1st correction, t0avg = mean time of oil release
c----------------------------------------------------------------------
      if(icrn.eq.1) then
        open(46,file=TRIM(experiment_folder)//'/xp_files/medslik.crn',
     &          status='old')
        read(46,'(a80)') empty
        read(46,'(i2)') ncrns
        do k=1,ncrns
          read(46,'(a80)') empty 
          read(46,'(i4)') kkk
          crntim(k)=dfloat(kkk)
          read(46,'(i2)') nvert 
          do j=1,nvert+1
            read(46,'(a80)') empty
          enddo         
          read(46,'(f6.3,3x,f6.3)') alat,alon
          xnew=xgrid(alon)
          ynew=ygrid(alat)                  
          read(46,'(f6.3,3x,f6.3)') alat,alon
          xold=xgrid(alon)
          yold=ygrid(alat)                   
          deltax=(xnew-xold)*delx
          deltay=(ynew-yold)*dely
          if(k.eq.1) then
            t0avg = tspill/2.d0
            if(crntim(1).lt.tspill) t0avg=crntim(1)/2.d0
              deltat=(crntim(1)-t0avg)*3600.d0
            endif
            if(k.ge.2) deltat=(crntim(k)-crntim(k-1))*3600.d0
            if(deltat.ne.0) then
              crnu(k)=deltax/deltat
              crnv(k)=deltay/deltat
              if(k.gt.1) then
                crnu(k)=crnu(k)+crnu(k-1)
                crnv(k)=crnv(k)+crnv(k-1)
              endif
          else
            write(6,*) 'Two spill corrections for the same times'
            write(6,*) 'Times are: ',crntim(k)
            stop
          endif
          read(46,'(a80)') empty 
        enddo
        close(46)
      endif 
c----------------------------------------------------------------------
c     ttn=thickness of thin slick (m) (= 10 microns)
c     ttki=initial thickness of thick slick (m) (= 2 cm)
c     afac=initial area ratio of thin to thick slick areas (afac=4-8)
c     compute initial areas, volumes & radii of thick and thin slicks
c----------------------------------------------------------------------
      ttn=0.00001d0
      ttki=0.02d0
      afac=4.0d0
      atk0=vmspl/(ttki+afac*ttn)
      atn0=atk0*afac
      vtk0=atk0*ttki
      vtn0=atn0*ttn
      rtk0=dsqrt(atk0/pi)
      rtn0=dsqrt((atn0+atk0)/pi)
      write(90,*) ' '
      write(90,*) 'initial mini-slick radii (thick/thin) = ',rtk0,rtn0
      write(90,*) ' '
c----------------------------------------------------------------------
c     initial assignment of mini-spill properties
c     tre = time from release
c     ttk/atk/vtk = thickness/area/volume of thick slick
c     ttn/atn/vtn = thickness/area/volume of thick slick
c     tt0/at0/vt0 = average thickness/total area/total volume of 2 slicks
c     den = average density of oil in minispill i
c     vis/visem = viscosity of oil/mousse in minispill i
c     vtke/vtne/vte = evaporated volumes in thick/thin/total slick
c     vtkd/vtnd/vtd = dispersed volumes in thick/thin/total slick
c     ftk/ftn = evaporated fractions
c     fw = water fraction in mousse
c     xcl/xcs = volumes of large/small oil droplets dispersed below slick
c     pcte/pctd = percentages evaporated/dispersed
c----------------------------------------------------------------------
      do i=1,nmini
        tre(i)=0.d0
        ttk(i)=ttki
        atk(i)=atk0
        atn(i)=atn0
        ato(i)=atk(i)+atn(i)
        vtk(i)=vtk0
        vtn(i)=vtn0
        vto(i)=vtk(i)+vtn(i)
        tto(i)=vto(i)/ato(i)
        den(i)=deno
        vis(i)=viso
        visem(i)=viso
        vtke(i) =0.d0
        vtne(i) =0.d0
        vte(i)  =0.d0
        vtkd(i) =0.d0
        vtnd(i) =0.d0
        vtd(i)  =0.d0
        ftk(i)  =0.d0
        ftn(i)  =0.d0
        fw(i)   =0.d0
        xcl(i)  =0.d0
        xcs(i)  =0.d0
        pcte(i) =0.d0
        pctd(i) =0.d0
        write(90,*) 'Initial minispill properties:'
        write(90,*) tre(i)
        write(90,*) ttk(i),atk(i),vtk(i) 
        write(90,*) ttn   ,atn(i),vtn(i) 
        write(90,*) tto(i),ato(i),vto(i) 
        write(90,*) ftk(i),ftn(i),fw(i) 
        write(90,*) ttk(i),atk(i),vtk(i) 
        write(90,*) vtke(i),vtne(i),vte(i) 
        write(90,*) pcte(i),pctd(i) 
      enddo
c----------------------------------------------------------------------
c     initial assignment of status parameter is for each parcel
c        is=0 parcel not released,
c        is=1 in the spreading surface slick 
c        is=2 on surface but not spreading
c        is=3 dispersed into water column
c        is=-nsg stuck on shore segment number nsg
c     ib(i) = k or -k indicates parcel is stuck on kth boom
c     c1p(i) = tons of evaporative component left in ith parcel
c     c2p(i) = tons of non-evaporative component left in ith parcel
c     px(i), py(i) = horizontal grid coordinates of ith parcel
c     pz(i) = vertical sigma-coordinate of ith parcel (0/1 on bottom/surface)
c----------------------------------------------------------------------
      ! Particles position
      if(isat.eq.1) then
        write(6,*) 'Initializing slick from contour'
        do j=1,nmini
          call readsat(ix,nppms,px_contour,
     &                 py_contour,experiment_folder)
          do i=1,nppms
            id_p = (j-1) * nppms + i
            px(id_p) = xgrid(px_contour(i))
            py(id_p) = ygrid(py_contour(i))
          enddo
        enddo
      elseif(isat.eq.0) then 
        do i=1,ntot
          ! new parcels randomly & uniformly distributed 
          ! within initial thick slick radius rtk0
          rnd = randmedslik(ix)
          radd = rtk0*dsqrt(rnd)
          phi = 2.d0*pi*randmedslik(ix)
          xds0 = radd*dcos(phi)/delx
          yds0 = radd*dsin(phi)/dely
          ! Particles initial position
          px(i) = x0 + xds0
          py(i) = y0 + yds0
        enddo
      endif
      ! Particles properties
      do i=1,ntot
        is(i) = 0
        bseg1_id(i) = -1
        bseg2_id(i) = -1
        ib(i) = 0
        c1p(i) = c1i*vpp
        c2p(i) = (1.d0-c1i)*vpp
        pz(i) = 1.d0
      enddo
c----------------------------------------------------------------------
c    open file for fate parameters and write headings
c----------------------------------------------------------------------
      if(irestart.eq.0) then
        open(99,file=TRIM(experiment_folder)//'/out_files/oil_fate.txt')
        write(99,600) idd,imm,iyr,istart
        write(99,601) splat,splon,x0,y0,tspill,icomp 
        if(ispill.gt.0) write(99,602) splrte,vlunit
        if(ispill.eq.0) write(99,603) splrte,vlunit
        write(99,680)              
  680   format('   time   vol spilt  %evap    %srf   %srftot   %disp',
     &       '  %cstfxd   %csttot   visem1   visem2  visoil1  visoil2',
     &       '  denoil1  denoil2   wfrac1  wfrac2  volratio')
      elseif(irestart.eq.1) then
        open(99,file=TRIM(experiment_folder)//'/out_files/oil_fate.txt',
     &        access='APPEND')
      endif
c----------------------------------------------------------------------
c     npcl = number of parcels released so far
c     nspill = number of minispills which have occured
c     nseg = number of coastal segments in vicinity of spill
c     totucrn,totvcrn = total observational corrections to u,v up to current time 
c----------------------------------------------------------------------
      npcl=nppms
      nspill=1
      last_release = .false.
      do i=1,npcl ! release initial minispill
        is(i)=1
      enddo
      nseg=0
      totucrn=0.d0
      totvcrn=0.d0
c----------------------------------------------------------------------
c     For hot restart read restart file
c----------------------------------------------------------------------
      if(irestart.eq.1) then
        if(iyr.ge.2000) write(ay,'(i2.2)') iyr-2000
        if(iyr.lt.2000) write(ay,'(i2.2)') iyr-1900
        write(am,'(i2.2)') imm
        write(ad,'(i2.2)') idd
        write(ah,'(i2.2)') istart/100
        write(a3,'(i3.3)') ihrestart
        open(98,file=TRIM(experiment_folder)//
     &          "/xp_files/"//ay//am//ad//ah//'_'//a3//'.rso',
     &          form='unformatted')

        read(98) jtime,npcl,nspill,pcevp,pcsrf,pcdsp,pccst,deno,viso
        if(jtime.ne.ihrestart) then
          write(6,*) 'Restart file time ',jtime,' does not agree ',
     &          'with restart time in input file ',ihrestart
          stop
        endif 
        do i=1,npcl
          read(98) is(i),ib(i),c1p(i),c2p(i),alon,alat,pz(i),seep(i),bseg1_id(i),bseg2_id(i) 
          px(i) = xgrid(alon)
          py(i) = ygrid(alat)
        enddo
        do i=1,nspill
          read(98) den(i),vis(i),visem(i),tre(i),c1ms(i),c2ms(i),
     &      atn(i),atk(i),ato(i),ttk(i),tto(i),
     &      vtn(i),vtk(i),vto(i),xcl(i),xcs(i),
     &      vtne(i),vtke(i),vte(i),vtnd(i),vtkd(i),vtd(i),
     &      ftk(i),ftn(i),fw(i),pcte(i),pctd(i) 
        enddo
        nsgr=1
    1   continue
        read(98,end=2) ns0(nsgr),xx1,yy1,vcst(nsgr)
        segx(nsgr)=(xx1-along1)/dlong+1.d0
        segy(nsgr)=(yy1-alatg1)/dlatg+1.d0 
        nsgr = nsgr + 1
        go to 1 
    2   continue
        close(98)
        nsgr = nsgr - 1        
        write(90,*) 'Impacted coastal segments '
        write(90,*) 'Num impacted segments = ',nsgr
        write(90,*) '    No Old-num Numpcls   Seep'
        do ns=1,nsgr
          num=0
          do i=1,npcl
            if(is(i).eq.-ns0(ns)) num=num+1
          enddo
          write(90,'(3i7,f13.6)') ns,ns0(ns),num,vcst(ns)
        enddo            
        totseep=0.
        do ns=1,nsgr
          totseep =totseep + vcst(ns)
        enddo
        write(90,*) 'total beached = ',totseep
        write(90,*) ' '

        pccst1=100.*(totseep)/(npcl*vpp)
        write(90,*) 'percentages beached = ',pccst,pccst1         
      endif
c----------------------------------------------------------------------
c     ADDING NETCDF DIMENSION AND VARIABLES
c----------------------------------------------------------------------
      NPARCEL=ntot
      NTIME=icomp/tiprs
c     Dimensions
      nc_status = nf_def_dim(ncid,'parcel_id', NPARCEL, prtcl_dimid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_def_dim(ncid, 'time', NTIME, time_dimid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
      dimids(1) = prtcl_dimid
      dimids(2) = time_dimid
c     Variables
      nc_status = nf_def_var(ncid, 'latitude', 5, NDIMS, dimids, 
     &          lat_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'longitude', 5, NDIMS, dimids, 
     &          lon_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'evaporative_volume', 6,  
     &          NDIMS, dimids,evol_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'non_evaporative_volume', 6,  
     &          NDIMS, dimids,nvol_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'water_fraction', 6,  
     &          NDIMS, dimids,wc_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'particle_status', 6, 
     &          NDIMS, dimids,prtt_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
      
      nc_status = nf_def_var(ncid, 'beach_seg_point1',4,
     &          ndims, dimids,bseg1_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
        
      nc_status = nf_def_var(ncid, 'beach_seg_point2',4,
     &          ndims, dimids,bseg2_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid,'time', 4, 1, (/time_dimid/), 
     &          time_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'seeped_vol', 5, 
     &          NDIMS, dimids, bv_varid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
c     SPILL VARIABLES
      nc_status = nf_def_var(ncid, 'total_fixed_oil',6,
     &          1, dimids(2),tfxd_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)    
      
      nc_status = nf_def_var(ncid, 'viscosity_emulsion_1',6,
     &          1, dimids(2),vemi_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'viscosity_emulsion_2',6,
     &          1, dimids(2),vemf_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'viscosity_oil_1',6,1,
     &             dimids(2),visi_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'viscosity_oil_2',6,1,
     &             dimids(2),visf_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'density_emulsion_1',6,1,
     &             dimids(2),deni_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'density_emulsion_2',6,1,
     &             dimids(2),denf_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'water_fraction_1',6,1,
     &             dimids(2),fwi_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'water_fraction_2',6,1,
     &             dimids(2),fwf_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_def_var(ncid, 'volume_ratio',6,1,
     &             dimids(2),volr_varid)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c     Assign units attributes to the netCDF variables.
      nc_status = nf_put_att_text(ncid, lat_varid, UNITS, 
     &          len(LAT_UNITS), LAT_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, lon_varid, UNITS,  
     &          len(LON_UNITS),LON_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, evol_varid, UNITS, 
     &          len(EVOL_UNITS),EVOL_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, nvol_varid, UNITS,  
     &          len(NVOL_UNITS),NVOL_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, wc_varid, UNITS,
     &          len(WC_UNITS), WC_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, prtt_varid, UNITS,
     &          len(PRTS_UNITS), PRTS_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, bseg1_varid, UNITS,
     &          len(BSEG1_UNITS), BSEG1_UNITS)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, bseg2_varid, UNITS,
     &          len(BSEG2_UNITS), BSEG2_UNITS)
      nc_status = nf_put_att_text(ncid, bv_varid, UNITS,
     &          len(BV_UNITS), BV_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, time_varid, UNITS,
     &          len(TIME_UNITS), TIME_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, tfxd_varid, UNITS,
     &          len(TFXD_UNITS), TFXD_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, vemi_varid, UNITS,
     &          len(VEM_UNITS), VEM_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, vemf_varid, UNITS,
     &          len(VEM_UNITS), VEM_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, visi_varid, UNITS,
     &          len(VIS_UNITS), VIS_UNITS)
      nc_status = nf_put_att_text(ncid, visf_varid, UNITS,
     &          len(VIS_UNITS), VIS_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, deni_varid, UNITS,
     &          len(DEN_UNITS), DEN_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, denf_varid, UNITS,
     &          len(DEN_UNITS), DEN_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, fwi_varid, UNITS,
     &          len(WC_UNITS), WC_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, fwf_varid, UNITS,
     &          len(WC_UNITS), WC_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_text(ncid, volr_varid, UNITS,
     &          len(VOLR_UNITS), VOLR_UNITS)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c     Other attributes - oil density and volume of each parcel
      nc_status = nf_put_att_double(ncid, evol_varid, OIL_DENSITY, 
     &          6,1,deno)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, evol_varid, PARCEL_VOL,  
     &          6,1,vpp)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

      nc_status = nf_put_att_double(ncid, nvol_varid, OIL_DENSITY, 
     &          6,1,deno)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, nvol_varid, PARCEL_VOL,  
     &          6,1,vpp)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, nvol_varid, SP_LON,  
     &          6,1,splon)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, nvol_varid, SP_LAT,  
     &          6,1,splat)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, evol_varid, SP_LON,  
     &          6,1,splon)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status = nf_put_att_double(ncid, evol_varid, SP_LAT,  
     &          6,1,splat)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c     Compressing variables
      nc_status=nf_def_var_deflate(ncid, lat_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status=nf_def_var_deflate(ncid,lon_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status=nf_def_var_deflate(ncid, evol_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status=nf_def_var_deflate(ncid, nvol_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status=nf_def_var_deflate(ncid, wc_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
      nc_status=nf_def_var_deflate(ncid, prtt_varid,0,1,4)
      if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c     End define mode.
      nc_status = nf_enddef(ncid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
c----------------------------------------------------------------------

c     begin the main computational loop;  nst=time step
c     timehr = time in hours from start of spill
c     tsec = time in seconds
c----------------------------------------------------------------------
      write(6,161)
      if(irestart.eq.1) write(6,162) ihrestart
       
  161 format(' Program initialization is now complete'/
     &       ' Spill simulation will now begin'/)
  162 format(' Run will restart from ',i4,' hours after spill'//)
c     MAIN LOOP (per timestep)
      do 94 nst=1,maxst
        timehr=nst*delt
        if(irestart.eq.1) timehr=timehr+ihrestart
        tsec=timehr*3600.d0
c----------------------------------------------------------------------
c select area that contains all the RELEASED parcels: (mzb,nzb)-(mzf,nzf) for 
c selection of coastal segments
c (xavg,yavg) approximates the grid coords of current centre of slick
c v3.00 wind is interpolated at the center of the AREA 
c containing all the particles
c----------------------------------------------------------------------
        xmin=10000.d0
        xmax=-10000.d0
        ymin=10000.d0
        ymax=-10000.d0
        do i=1,npcl
          if (is(i).ne.0) then
            if(px(i).lt.xmin) xmin=px(i)
            if(px(i).gt.xmax) xmax=px(i)
            if(py(i).lt.ymin) ymin=py(i)
            if(py(i).gt.ymax) ymax=py(i)
          endif
        enddo
c       compute center of area containing the slick
        mzb=int(xmin)-2
        mzf=int(xmax+1)+2
        nzb=int(ymin)-2
        nzf=int(ymax+1)+2
        if(mzb.lt.1) mzb=1
        if(nzb.lt.1) nzb=1
        if(mzf.gt.mm) mzf=mm
        if(nzf.gt.nm) nzf=nm
        xavg=dfloat(mzb+mzf)/2.d0
        yavg=dfloat(nzb+nzf)/2.d0
        m1=int(xavg+0.5d0)
        n1=int(yavg+0.5d0)
c       read forecast currents
        call fcstcur_1hr(nst,timehr+tstart,delt,nfcst,
     &      fcstfn,ifcstfn,fcsttim,
     &      fcstcurdir,sst,wx,
     &      wy,us,vs,u10,v10,
     &      u30,v30,u120,v120,dsst,
     &      dwx,dwy,dus,dvs,
     &      du10,dv10,du30,dv30,
     &      du120,dv120,itype) 

c----------------------------------------------------------------------
c     select water current field to be used for slick transport; 
c----------------------------------------------------------------------
        if(idepth.eq.fcstdep1) then
          uadv(:,:)=u10(:,:)
          vadv(:,:)=v10(:,:)
        elseif(idepth.eq.fcstdep2) then
          uadv(:,:)=u30(:,:)
          vadv(:,:)=v30(:,:)
        elseif(idepth.eq.fcstdep3) then
          uadv(:,:)=u120(:,:)
          vadv(:,:)=v120(:,:)
        else
          uadv(:,:)=us(:,:)
          vadv(:,:)=vs(:,:)
        endif

c----------------------------------------------------------------------
c     compute wind speed and direction
c  (winx,winy) = wind velocity components (m/s)
c  (wx,wy) = wind velocity that was used to compute forecast 
c         currents. This is already fixed in subroutine fcstcur
c     set evaporation rate parameter that depends on wind speed.
c----------------------------------------------------------------------
  
        call ski_ecmwf_wind(xavg,yavg,nst,timehr+tstart,delt,
     &             nwfcst,wfcstfn,iwfcstfn,wfcsttim,winx,winy,
     &             wvel,wdir,dwinx,dwiny,preproc_folder,itype)
        ce1=akew*(wvel*3.6d0)**gamma_evap
c----------------------------------------------------------------------
c     compute the wind-induced drift velocity for the oil parcels.
c     slick speed = alpha * wind speed at angle beta to right of wind.
c     if ibetared = 1, beta reduces with wind velocity, else const. 
c      subtract a fraction wredfrac of the wind already incorporated in 
c     the forecast current in case when iwindred = 1.
c----------------------------------------------------------------------
        if(wvel.eq.0.d0) then
              beta=beta0
            else
              beta=beta0*(1.d0-ibetared*wvel/(wvel+halfspeed))
            endif 
        csbeta=dcos(beta/degrad)
        snbeta=dsin(beta/degrad)

        do m=1,mm
          do n=1,nm
            winu=winx(m,n) - iwindred*wredfrac*wx(m,n)
            winv=winy(m,n) - iwindred*wredfrac*wy(m,n)
            wdrftx(m,n)=alpha*( winu*csbeta+winv*snbeta)
            wdrfty(m,n)=alpha*(-winu*snbeta+winv*csbeta)
          enddo
        enddo

   
c----------------------------------------------------------------------
c         Stokes drift velocity calculation
c         using JONSWAP spectrum parameterization 
c         (M. De Dominicis)
c---------------------------------------------------------------------- 
        if(istoke.eq.0) then
          do m=1,mm
            do n=1,nm
                stoku(m,n)=0.d0
                stokv(m,n)=0.d0
            enddo
          enddo
        elseif(istoke.eq.1) then
          wdirstoke=wdir
          call calcfetch(xavg,yavg,wdirstoke,fetch,preproc_folder)
          xavg_lon=glon(xavg)
          yavg_lat=glat(yavg)
          grav=9.8
          pi=4.*datan(1.d0)
          wxx=winx(m1,n1)
          wyy=winy(m1,n1)
          wvel=dsqrt(wxx*wxx+wyy*wyy)
          wwdir=0.d0
          if(wxx.eq.0.) then
            wwdir=0.d0
            if(wyy.gt.0) wwdir=180.d0
            if(wyy.le.0) wwdir=0.d0
          else
            wwdir=datan(wyy/wxx) * degrad
            if(wxx.lt.0.d0) wwdir=wwdir+180.d0
            wwdir=270.d0-wwdir
          endif         
          do m=1,mm
            do n=1,nm
              ww=dsqrt((winx(m,n)*winx(m,n))+(winy(m,n)*winy(m,n))) !wind velocity module     
              if(ww.eq.0)then
                stoke=0
                stoku(m,n)=0
                stokv(m,n)=0
              else
                gamma=3.3
                alfa=0.076*((ww**2)/(fetch*grav))**(0.22)
                fang_peak=22*((grav**2)/(ww*fetch))**(0.3333333333)
                freq_sp(1)=0.001
                do k=2,700
                  freq_sp(k)=freq_sp(k-1)+0.001
                  fang(k)=2*pi*freq_sp(k)
                  if (fang(k).ge.fang_peak) sigma=0.09
                  if (fang(k).lt.fang_peak) sigma=0.07
                  erre(k)=exp(-((fang(k)-fang_peak)**2)
     &                   /(2*(sigma**2)*(fang_peak**2)))
                  sp_exp2(k)=exp(-1.25*(fang_peak/fang(k))**4)
                  sp_exp1(k)=alfa*grav**2*fang(k)**(-5)
                  spectra(k)=sp_exp1(k)*sp_exp2(k)*(gamma**erre(k))
                  wave_num(k)=fang(k)*fang(k)/grav
                  stoke_sp(k)=2*spectra(k)*fang(k)*wave_num(k)
                enddo
                hwave_tot=0
                stoke_tot=0
                do k=1,699
                  hwave_d(k)=0.001*2*pi*(spectra(k)+spectra(k+1))/2
                  stoke_d(k)=0.001*2*pi*(stoke_sp(k)+stoke_sp(k+1))/2
                  stoke_tot=stoke_tot+stoke_d(k)
                  hwave_tot=hwave_tot+hwave_d(k)
                enddo
                hwave=4*sqrt(hwave_tot)
                stoke=stoke_tot
                wwangle = (270.d0-wwdir) / degrad
                cswwdir = dcos(wwangle)
                snwwdir = dsin(wwangle)
                stoku(m,n)=stoke*cswwdir 
                stokv(m,n)=stoke*snwwdir
              endif
            enddo
          enddo
        endif
       
c----------------------------------------------------------------------
c     compute sea surface temperature
c     tc, tk = sea surface temp in deg centigrade and kelvin
c
c     set evap and disp properties that depend on sst
c     denw = density of sea water at temp tk
c     viso = oil viscosity at temp tk; tvk0 = ref temp for initial viscosity
c     re-initialize minispill oil properties that depend on temp
c     write initial data into fate output file for time 0
c----------------------------------------------------------------------    
        tc=sst(m1,n1)
        tk=tc+273.d0
        if(nst.eq.1.and.irestart.eq.0) then
          fac = dexp(cvt*((1.d0/tk)-(1.d0/tvk0)))
          viso = viso * fac
          do i=1,nmini
            den(i)=deno
            vis(i)=viso
            visem(i)=viso
          enddo
          timout=0.
          ntons=0.d0
          if(ispill.eq.0) ntons = totvol
          pcevap   =0.d0
          pcsrf    =100.d0
          pcsrftot  =100.d0
          pcdsp    =0.d0
          pccstfxd =0.d0
          pccsttot =0.d0
          wfr      =0.d0
          volratio =1.d0
          write(99,'(f9.2,f9.0,6f9.4,6f9.2,2f8.3,f10.5)') 0.d0,
     &        ntons,pcevp,pcsrf,pcsrftot,pcdsp,pccstfxd,pccsttot,
     &        visem(1),viso,viso,viso,deno,deno,wfr,wfr,volratio
        endif
c----------------------------------------------------------------------
c     compute corrections to oil drift velocity from slick observation
c----------------------------------------------------------------------
        if(icrn.eq.1) then
          if(timehr.le.crntim(1)) k1=1
          do k=2,ncrns
            if(timehr.le.crntim(k).and.timehr.gt.crntim(k-1)) k1=k
          enddo
          if(timehr.gt.crntim(ncrns)) k1=ncrns
          totucrn=crnu(k1)
          totvcrn=crnv(k1)
        endif
c----------------------------------------------------------------------
c     Using subroutine coast: reads coast segments and coastal types and 
c     selects those segments that lie within the spill region
c     
c     alngt(i) = length of coastal segment i in metres
c     prel(i) = prob of oil being released in interval delt
c     sfrac(i) = frac of oil seeping into segment i in delt
c
c     For restart locate beached parcels on their new coastal segments
c----------------------------------------------------------------------
        call coast(delt,seg,sfrac,prel,nseg,alngt,
     &        api,apicoeff,preproc_folder,itype,bseg1_pid,bseg2_pid)
        if(irestart.eq.1.and.nst.eq.1) then
          write(90,*) 'Reassign beached parcels to ',nseg,' new segmnts'
          do ns=1,nss
            seep(ns) = 0.d0
          enddo
          do i=1,npcl
            itmp(i) = 0
            itmp_bseg1_id(i) = -1
            itmp_bseg2_id(i) = -1
          enddo
          
          do n=1,nsgr
            dmin=100000.d0
            do ns=1,nseg
              dx = segx(n) - seg(ns,1)
              dy = segy(n) - seg(ns,2)
              d = dx*dx + dy*dy
              if(d.lt.dmin) then
                dmin = d
                nsmin = ns
              endif
            enddo
            seep(nsmin) = vcst(n)
            num=0
            do i=1,npcl
              if(is(i).eq.-ns0(n)) then
                itmp(i) = -nsmin
                itmp_bseg1_id(i) = bseg1_id(ns)
                itmp_bseg2_id(i) = bseg2_id(ns)
                num=num+1
              endif
            enddo
          enddo
          do i=1,npcl
            if(is(i).lt.0) then
              is(i) = itmp(i)
              bseg1_id(i) = itmp_bseg1_id(ns)
              bseg2_id(i) = itmp_bseg2_id(ns)
            endif
          enddo
          write(90,*)
          write(90,*) 'New1y impacted coastal segments '
          write(90,*) 'Segment Numpcls Seep'

          totseep = 0.d0
          do ns=1,nss
            if(seep(ns).gt.0.d0) then
              totseep =totseep + seep(ns)
              num=0
              do i=1,npcl
                if(is(i).eq.-ns) num=num+1
              enddo
              write(90,'(2i7,f13.6)') ns,num,seep(ns)
            endif
          enddo
          write(90,*) 'total beached 1 = ',totseep
          write(90,*) ' '
        endif
c----------------------------------------------------------------------
c     release new mini-spill
c----------------------------------------------------------------------
        if (nspill.lt.nmini) then
          npcl0=npcl
          nspill=nspill+1
          npcl=npcl+nppms
          do i=npcl0+1,npcl
            is(i)=1
          enddo
        else
          last_release = .true.
        endif
        en1ps=0.d0
        en2ps=0.d0
        en3ps=0.d0
        en4ps=0.d0
        en5ps=0.d0
        en6ps=0.d0
        tvolsrf=0.d0
c----------------------------------------------------------------------
c     start mini spill loop
c     mini-spill ns contains parcels l1 through l2
c     tre(ns): time in hours since release of mini spill ns
c     nsps: no of parcels still spreading from mini spill ns
c     xcm, ycm: centre of spreading part of slick mini spill ns
c     xcmall, ycmall: centre of whole mini spill ns
c----------------------------------------------------------------------
        if(nst.eq.1.and.irestart.eq.0) then
          write(90,*) 
          write(90,*) 'Release of parcels:'
        endif
        if(timehr.le.tspill+delt) write(90,*) 'Time hrs = ',timehr

        do 70 ns=1,nspill
          l1=(ns-1)*nppms+1
          l2=ns*nppms
          if (ns.eq.nspill.and.(.not.last_release)) go to 73
          tre(ns)=tre(ns)+delt
          nsps=0
          c1ms(ns)=0.d0
          c2ms(ns)=0.d0
          xcm=0.d0
          ycm=0.d0
          nspsall = 0
          xcmall = 0.d0
          ycmall = 0.d0
          if(timehr.le.tspill+delt) then 
            write(90,*) '    Minispill no. ',ns
            write(90,*) '    Parcels ',l1,' to ',l2
          endif
c----------------------------------------------------------------------
c     stop minispill spreading after time sprdmx
c     compute centres of spreading part and whole mini-slick
c     compute Smagorinsky diffusivity at centre of each mini-slick
c----------------------------------------------------------------------
          do 58 i=l1,l2
            if (tre(ns).ge.sprdmx.and.is(i).eq.1) then
              is(i)=2
              bseg1_id(i)=-1
              bseg2_id(i)=-1
            endif
            if (is(i).eq.1) then
              xcm=xcm+px(i)
              ycm=ycm+py(i)
              nsps=nsps+1
            endif

            xcmall = xcmall + px(i)
            ycmall = ycmall + py(i)
            nspsall = nspsall + 1
   58     continue

          if (nsps .ge. 1) then
            xcm=xcm/nsps
            ycm=ycm/nsps
          else
            xcm = x0
            ycm = y0
          endif

          if (nspsall .ge. 1) then
            xcmall = xcmall / nspsall 
            ycmall = ycmall / nspsall 
          else
            xcmall = xavg
            ycmall = yavg
          endif
c
          if(ismag.eq.1) then
            call smag(xcmall,ycmall,us,vs,horizk,itype)
            horizd=sqrt(6.d0*horizk*delt*3600.d0)
          endif   
c----------------------------------------------------------------------
c     Using subroutine ed: computes evaporating and dispersing fractions
c     and rate of mousse formation; called ned times per step
c     evaporation, dispersion and emulsification by mckay
c     pctd,probd = percent dispersed and probability of disperion
c     pcte,frace = percent and fraction evaporated
c     fw(ns) = fraction of water in minispill ns
c     rratio = ratio of radii of thick slick before and after time step
c----------------------------------------------------------------------
          pctd0=pctd(ns)
          atkns0 = atk(ns)
          rtkns0 = dsqrt(atk(ns) / pi) 
          do k=1,ned 
            call ed(tedsec,vmspl,den(ns),vis(ns),
     &         visem(ns),ttk(ns),ttn,
     &         tto(ns),atk(ns),atn(ns),ato(ns),vtk(ns),vtn(ns),vto(ns),
     &         ftk(ns),ftn(ns),ft,fw(ns),
     &         vtke(ns),vtne(ns),vte(ns),vtkd(ns),vtnd(ns),vtd(ns),
     &         xcl(ns),xcs(ns),pcte(ns),pctd(ns))
          enddo       
c       Aged spill properties R. Lardner and M. De Dominicis (2009)
          if(isat.eq.1.and.nst.eq.nst0) then 
            write(90,*) 'Aged minispill properties:'
            write(90,*) tre(ns)
            write(90,*) ttk(ns),atk(ns),vtk(ns) 
            write(90,*) ttn   ,atn(ns),vtn(ns) 
            write(90,*) tto(ns),ato(ns),vto(ns) 
            write(90,*) ftk(ns),ftn(ns),fw(ns) 
            write(90,*) ttk(ns),atk(ns),vtk(ns) 
            write(90,*) vtke(ns),vtne(ns),vte(ns) 
            write(90,*) pcte(ns),pctd(ns) 
          endif
          probd=(pctd(ns)-pctd0)/(100.d0-pctd0)
          frace=pcte(ns)/100.d0
          rtkns = dsqrt(atk(ns) / pi) 
          rratio = rtkns / rtkns0
c----------------------------------------------------------------------
c     displace and transform the lagrangian parcels
c     first evaporation and dispersion check
c----------------------------------------------------------------------
          do 66 i=l1,l2
            c1p(i)=(c1i-frace)*vpp
            if(isat.eq.1.and.nst.le.nst0) go to 65
            if(is(i).eq.9.or.is(i).eq.4) go to 65
            rrr=randmedslik(ix)
            if((is(i).eq.1.or.is(i).eq.2).and.rrr.lt.probd) then
              is(i)=3
              bseg1_id(i)=-1
              bseg2_id(i)=-1
            endif
c----------------------------------------------------------------------
c     compute spreading displacement (thick slick contains most of the oil
c       since mechanical spreading is stopped after sprdmx = 24 hrs)
c----------------------------------------------------------------------
            xds=0.d0
            yds=0.d0
            if(is(i).eq.1) then
                if (rtkns0.gt.0.d0) then
                  xds = (rratio - 1.d0) * (px(i) - xcm)
                  yds = (rratio - 1.d0) * (py(i) - ycm)
                endif
            endif
c----------------------------------------------------------------------
c     compute diffusion displacement
c----------------------------------------------------------------------
            xdd=(2.d0*randmedslik(ix)-1.d0)*horizd
            ydd=(2.d0*randmedslik(ix)-1.d0)*horizd
            zdd = 0.d0
c----------------------------------------------------------------------
c     compute advective displacements of surface & dispersed parcels
c     include wind drift & correction term from observation(s) of spill
c----------------------------------------------------------------------
c     Bilinear interpolation for wind & stoke drift velocity (added by 
c     M. De Dominicis) 
c---------------------------------------------------------------------
            ui = 0.d0
            vi = 0.d0
            if(is(i).ne.3) then
              call intrpl(px(i),py(i),uadv,itype,ui)
              call intrpl(px(i),py(i),vadv,itype,vi)
              call intrpl(px(i),py(i),wdrftx,itype,wui)
              call intrpl(px(i),py(i),wdrfty,itype,wvi)
              call intrpl(px(i),py(i),stoku,itype,sui)
              call intrpl(px(i),py(i),stokv,itype,svi)
              ui=ui+sui+wui
              vi=vi+svi+wvi
            else if(is(i).eq.3) then
              pxi=int(px(i)+0.5d0)
              pyi=int(py(i)+0.5d0)
              call intrpl0(px(i),py(i),h,hint)
              dep=(1.d0-pz(i))*hint
              zdd=(2.d0*randmedslik(ix)-1.d0)*
     &              vertd(dep,vertd1,vertd2,thermocl)
              if(dep.le.fcstdep1) then
                ui=(us(pxi,pyi)*(fcstdep1-dep)+u10(pxi,pyi)*dep)/10.d0
                vi=(vs(pxi,pyi)*(fcstdep1-dep)+v10(pxi,pyi)*dep)/10.d0
              else if(dep.gt.fcstdep1.and.dep.le.fcstdep2) then
                if(hint.ge.fcstdep2) then
                  denom=fcstdep2-fcstdep1
                  fac1=(dep-fcstdep1)
                  fac2=(fcstdep2-dep) 
                else
                  denom=hint-fcstdep1
                  fac1=(dep-fcstdep1)
                  fac2=(hint-dep) 
                endif
                ui=(u30(pxi,pyi)*fac1+u10(pxi,pyi)*fac2)/denom
                vi=(v30(pxi,pyi)*fac1+v10(pxi,pyi)*fac2)/denom
              else if(dep.gt.fcstdep2.and.dep.le.fcstdep3) then
                if(hint.ge.fcstdep3) then
                  denom=fcstdep3-fcstdep2
                  fac1=(dep-fcstdep2)
                  fac2=(fcstdep3-dep) 
                elseif(hint.lt.fcstdep2) then
                  denom=1.d0
                  fac1=0.d0
                  fac2=1.d0
                else
                  denom=hint-fcstdep2
                  fac1=(dep-fcstdep2)
                  fac2=(hint-dep) 
                endif
                ui=(u120(pxi,pyi)*fac1+u30(pxi,pyi)*fac2)/denom
                vi=(v120(pxi,pyi)*fac1+v30(pxi,pyi)*fac2)/denom
              elseif(dep.gt.fcstdep3) then
                ui=u120(pxi,pyi)
                vi=v120(pxi,pyi)
              endif
            endif
            xdc=(ui+totucrn)*delt*3600.d0
            ydc=(vi+totvcrn)*delt*3600.d0
c----------------------------------------------------------------------
c     displace parcels on surface and dispersed parcels
c     (m21,n21) = old grid coordinates on bathy grid (nearest grid point)
c     (ppx,ppy) = new grid coordinates of parcel i
c----------------------------------------------------------------------
            m21=int(px(i)+0.5d0)
            n21=int(py(i)+0.5d0)
            xdispl = (xds+xdc+xdd)
            ydispl = (yds+ydc+ydd)
            ppx = px(i) + xdispl/delx
            ppy = py(i) + ydispl/dely
c----------------------------------------------------------------------
c     check if beached parcels are released
c     sfrac = fraction of beached oil seeping into sand per step at 'ns'
c     prel = probability of beached oil being washed off at 'ns'
c     seep(nsg) = volume of oil at the impacted coastal segment 'nsg'
c     seepage slows when it approaches carrying capacity seepmx (tons/km)
c     (xdispl,ydispl) must be to left of (dxseg,dyseg)
c----------------------------------------------------------------------
            if(is(i).lt.0.and.ib(i).eq.0) then
              nsg=-is(i)
              seepge=c2p(i)*sfrac(nsg)
              fac = seep(nsg) / ( seepmx * alngt(nsg) )
              if(fac.gt.10.d0) write(90,*) nst,i,fac
              reducn = dexp( - fac )
              seepge = seepge * reducn
              c2p(i)=c2p(i)-seepge
              seep(nsg)=seep(nsg)+seepge
              probr=prel(nsg)
              dxseg = seg(nsg,3) - seg(nsg,1)
              dyseg = seg(nsg,4) - seg(nsg,2)
              cross = dxseg * (ppy - py(i)) - dyseg * (ppx - px(i))
              if((randmedslik(ix).lt.probr).and.cross.gt.0.d0) then
                pz(i)=1.d0
                is(i)=2
                bseg1_id(i)=-1
                bseg2_id(i)=-1
              endif
            endif
c----------------------------------------------------------------------
c     check if surface parcel hits one of the booms 
c     ib(i) = +k indicates parcel is on right side of boom k
c     ib(i) = -k indicates parcel is on left side of boom k
c----------------------------------------------------------------------
            if((is(i).eq.1.or.is(i).eq.2).and.ib(i).eq.0) then
              xi1 = px(i)
              eta1 = py(i)
              xi2 = ppx
              eta2 = ppy
              dmin = 100.d0
              nbmin=0
              ibmin=0
              do 64 k=1,nbooms
                if(timehr+tstart.lt.bmtim(k)) go to 64
                xx1 = bmx1(k)
                yy1 = bmy1(k)
                xx2 = bmx2(k)
                yy2 = bmy2(k)
                ddel  = (xx2-xx1)*(eta2-eta1)-(yy2-yy1)*(xi2-xi1)
                ddel1 = (eta2-eta1)*(xi1-xx1)-(xi2-xi1)*(eta1-yy1)
                ddel2 = (yy2-yy1)*(xi1-xx1)-(xx2-xx1)*(eta1-yy1)

                if(ddel.eq.0.d0) go to 64
                if( 100*randmedslik(ix).gt.ibmeff(k) ) go to 64

                alam=ddel1/ddel
                alamp=ddel2/ddel
                if(alam.ge.0.d0.and.alam.le.1.d0.and.
     &                       alamp.ge.0.d0.and.alamp.le.1.d0) then
                  xx=xx1+alam*(xx2-xx1)
                  yy=yy1+alam*(yy2-yy1)
                  dd1=dabs(xx-xi1)+dabs(yy-eta1)
                  if(dd1.lt.dmin) then
                    dmin=dd1
                    pppx=xx
                    pppy=yy
                    nbmin=k                  
                    if(ddel.gt.0.d0) ibmin=1
                    if(ddel.lt.0.d0) ibmin=-1
                  endif  
                endif
   64         continue             
              if(nbmin.ne.0) then
                px(i)=pppx
                py(i)=pppy
                ib(i)=ibmin*nbmin
              endif
            endif
c----------------------------------------------------------------------
c     vertical diffusion of dispersed parcels
c     ppz = temporary new sigma coordinate of parcel (1 at surface)
c     if ppz > 1 reflect displacement from surface
c     if ppz*caph < 0.2 (< 20 cm from bottom) parcel is sedimented
c----------------------------------------------------------------------
            if(is(i).eq.3) then
              caph=h(m21,n21)
              ppz=1.0d0
              if(caph.gt.0.2d0) ppz=pz(i)+zdd/caph
              if(ppz.gt.1.) ppz=2. - ppz
              if(ppz*caph.lt.0.2) then
                ppz=0.
                is(i)=4
                bseg1_id(i)=-1
                bseg2_id(i)=-1
              endif   
              pz(i)=ppz
            endif
c----------------------------------------------------------------------
c     check if parcel hits any coastal segment
c     if parcel 'i' hits coastal segment 'ns', set is(i) = -ns
c----------------------------------------------------------------------
            if(is(i).gt.0.and.is(i).ne.3.and.ib(i).eq.0) then
              xi1 = px(i)
              eta1 = py(i)
              xi2 = ppx
              eta2 = ppy
              dmin = 100.d0
              nsmn=0      
              do 62 nsg=1,nseg
                xx1 = seg(nsg,1)
                yy1 = seg(nsg,2)
                xx2 = seg(nsg,3)
                yy2 = seg(nsg,4)
                ddel  = (xx2-xx1)*(eta2-eta1)-(yy2-yy1)*(xi2-xi1)
                ddel1 = (eta2-eta1)*(xi1-xx1)-(xi2-xi1)*(eta1-yy1)
                ddel2 = (yy2-yy1)*(xi1-xx1)-(xx2-xx1)*(eta1-yy1)
                if(ddel.eq.0.d0) go to 62  
                alam=ddel1/ddel
                alamp=ddel2/ddel
                if(alam.ge.0.d0.and.alam.lt.1.d0.and.
     &                       alamp.ge.0.d0.and.alamp.lt.1.d0) then
                  xx=xx1+alam*(xx2-xx1)
                  yy=yy1+alam*(yy2-yy1)
                  dd1=dsqrt( (xx-xi1)*(xx-xi1) + (yy-eta1)*(yy-eta1) )
                  if(dd1.lt.dmin) then
                    dmin=dd1
                    pppx=xx
                    pppy=yy
                    nsmn=nsg
                    bsegxxyy1_id = bseg1_pid(nsg)
                    bsegxxyy2_id = bseg2_pid(nsg)
                  endif  
                endif
   62         continue
c             Update particles position (except for sedimented ones)
              if(nsmn.ne.0) then
                px(i)=pppx
                py(i)=pppy  
                if((px(i).le.0).or.(px(i).ge.mm).or.
     &             (py(i).le.0).or.(py(i).ge.nm)) then
                    is(i)=9
                    bseg1_id(i)=0
                    bseg2_id(i)=0
                else
                ! beached particle
                    is(i)=-nsmn
                    bseg1_id(i)=bsegxxyy1_id
                    bseg2_id(i)=bsegxxyy2_id
                endif
              else
                px(i)=ppx
                py(i)=ppy
                if((px(i).le.0).or.(px(i).ge.mm).or.
     &            (py(i).le.0).or.(py(i).ge.nm)) then
                  is(i)=9
                  bseg1_id(i)=0
                  bseg2_id(i)=0
                endif
              endif
            endif
c----------------------------------------------------------------------
c     count fractions of light and heavy components left in minispill ns
c----------------------------------------------------------------------
   65       continue ! Jump point if satellite data or sedimented/outside particles
   66     continue ! end particles loop
   73     continue
c----------------------------------------------------------------------
c     compute totals for output:
c         en1ps = vol of oil still spreading 
c         en2ps = vol of oil on surface but no longer spreading 
c         en3ps = vol of oil dispersed 
c         en4ps = vol of oil on the coast but not permanently there
c         en5ps = vol of oil sedimented
c         en6ps = vol of oil out of boundaries
c         tvolsrf = total vol of oil-water mousse incl releasable oil on coast
c----------------------------------------------------------------------
          do i=l1,l2
            c1ms(ns)=c1ms(ns)+c1p(i)
            c2ms(ns)=c2ms(ns)+c2p(i)
            if(is(i).eq.1) en1ps=en1ps+c1p(i)+c2p(i)
            if(is(i).eq.2) en2ps=en2ps+c1p(i)+c2p(i)
            if(is(i).eq.3) en3ps=en3ps+c1p(i)+c2p(i)
            if(is(i).lt.0) en4ps=en4ps+c1p(i)+c2p(i)
            if(is(i).eq.4) en5ps=en5ps+c1p(i)+c2p(i)
            if(is(i).eq.9) en6ps=en6ps+c1p(i)+c2p(i)
            if(is(i).ne.0.and.is(i).le.2) 
     &                tvolsrf=tvolsrf+(c1p(i)+c2p(i))/(1.d0-fw(ns))
          enddo
   70   continue ! end minispill loop
        volratio=tvolsrf/(npcl*vpp)
c----------------------------------------------------------------------
c     print spill output. first summary statistics of fate percentages
c     ttseep = vol of oil permanently on the coast -> pccstfxd = %
c     pcsrftot & tvolsrf include oil on coast but not permanently there
c     pcsrf excludes such oil
c     pccsttot includes all oil on coast both permanent and temporary
c     pccstfxd incudes only oil permanently attached to coast
c----------------------------------------------------------------------
        ttseep=0.d0
        do nsg=1,nseg
          ttseep=ttseep+seep(nsg)
        enddo
        c1tot=0.d0
        do ns=1,nspill
          c1tot=c1tot+c1ms(ns)
        enddo
        c1tot = c1tot/(npcl*vpp)
        pcevp    =100.d0*(c1i-c1tot)
        pcsrftot =100.d0*(en1ps+en2ps+en4ps+en6ps)/(npcl*vpp)
        pcsrf    =100.d0*(en1ps+en2ps+en6ps)/(npcl*vpp)
        pcdsp    =100.d0*(en3ps+en5ps)/(npcl*vpp)
        pccstfxd =100.d0*(ttseep)/(npcl*vpp)
        pccsttot =100.d0*(en4ps+ttseep)/(npcl*vpp)
        ntons=npcl*vpp
c----------------------------------------------------------------------
c     ADDING VALUES TO NC FILE
c----------------------------------------------------------------------  
c     These settings tell netcdf to write one timestep of data. (The
c     setting of start(2) inside the loop below tells netCDF which
c     timestep to write.)
        if(nst.eq.nprs) then
          ncounter(1) = 1
          ncounter(2) = 1
          ncpointer(2)=nst/nstph
          do i=1,ntot
            ncpointer(1)=i
            x_coordinate=glon(dble(px(i)))
            y_coordinate=glat(dble(py(i)))
            if(is(i).ge.0) then 
              bseg1_id(i)=-1
              bseg2_id(i)=-1
            endif
            nc_time=int(nst/nstph)
c    PARTICLES

            nc_status = nf_put_vara_real(ncid, lat_varid, ncpointer,
     &        ncounter,y_coordinate)
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

            nc_status = nf_put_vara_real(ncid, lon_varid, ncpointer,
     &        ncounter,x_coordinate)
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

            nc_status = nf_put_vara_double(ncid, evol_varid, ncpointer,
     &        ncounter,c1p(i))
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

            nc_status = nf_put_vara_double(ncid, nvol_varid, ncpointer,
     &        ncounter,c2p(i))
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

            nc_status = nf_put_vara_double(ncid, wc_varid, ncpointer,
     &        ncounter,fw(ns))
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c  PARTICLE STATUS and COAST SEGMENT 
            nc_status = nf_put_vara_double(ncid, prtt_varid, ncpointer,
     &        ncounter,dble(is(i)))
            if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
c BEACHED VOLUME  
            if(is(i).ge.0) then
              nc_status = nf_put_vara_real(ncid, bv_varid, ncpointer,
     &        ncounter,0.0)  ! no segment if the particle is not beached
            else
              nc_status = nf_put_vara_real(ncid, bv_varid, ncpointer,
     &        ncounter,seep(-int(is(i))))
            endif
            if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)

c BEACHED SEGMENT EXTREME 1  
        nc_status = nf_put_vara_int(ncid, bseg1_varid, 
     &                        ncpointer,ncounter,bseg1_id(i))
        if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
c BEACHED SEGMENT EXTREME 2  
        nc_status = nf_put_vara_int(ncid, bseg2_varid,
     &                        ncpointer,ncounter,bseg2_id(i))
        if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)
          enddo
c  TIME
          nc_status = nf_put_vara_int(ncid, time_varid,ncpointer(2),
     &        ncounter(2),nc_time)
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
c  SPILL PROPERTIES
        nc_status = nf_put_vara_double(ncid,tfxd_varid,
     &        ncpointer(2),ncounter(2),pccstfxd)
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, vemi_varid,ncpointer(2),
     &        ncounter(2),visem(1))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, vemf_varid,ncpointer(2),
     &        ncounter(2),visem(nmini))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)  
          nc_status = nf_put_vara_double(ncid, visi_varid,ncpointer(2),
     &        ncounter(2),vis(1))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, visf_varid,ncpointer(2),
     &        ncounter(2),vis(nmini))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, deni_varid,ncpointer(2),
     &        ncounter(2),den(1))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, denf_varid,ncpointer(2),
     &        ncounter(2),den(nmini))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, fwi_varid,ncpointer(2),
     &        ncounter(2),fw(1))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, fwf_varid,ncpointer(2),
     &        ncounter(2),fw(nmini))
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
          nc_status = nf_put_vara_double(ncid, volr_varid,ncpointer(2),
     &        ncounter(2),volratio)
          if (nc_status .ne. nf_noerr) call handle_nf_err(nc_status)
        endif
c----------------------------------------------------------------------
c    write fate parameters
c       time  volrel  %evap   %srf  %disp   %cst  visc1   visc2
c----------------------------------------------------------------------
        write(99,'(f9.2,f9.0,6f9.4,6f9.2,2f8.3,f10.5)') timehr,
     &     ntons,pcevp,pcsrf,pcsrftot,pcdsp,pccstfxd,pccsttot,
     &     visem(1),visem(nmini),vis(1),vis(nmini),
     &     den(1),den(nmini),fw(1),fw(nmini),volratio
        if(.not.(nst.lt.nprs)) then
          jtime=timehr+0.001d0
          write(a4,'(i4.4)') jtime
          do nsg=1,nseg
            vcst(nsg)=0.d0
            if(.not.(seep(nsg).eq.0.d0)) then
              dist=alngt(nsg)
              vcst(nsg)=seep(nsg)/dist  
            endif
          enddo
          nprs=nprs+iprs
        endif
        jtime=nst/nstph
        if(jtime*nstph.eq.nst) then
          if(irestart.eq.0) then
            write(6,*) 'Simulation has now completed ',jtime,' hours'
          else 
            write(6,*) 'Simulation has now completed ',jtime,' hours',
     &               ' after restarting'
          endif
        endif
c----------------------------------------------------------------------
c     next step !!!
c----------------------------------------------------------------------
   94 continue
      write(6,*) 'Simulation has completed successfully'
c     Close the file. This causes netCDF to flush all buffers and make
c     sure your data are really written to disk.
      nc_status = nf_close(ncid)
      if (nc_status.ne.nf_noerr) call handle_nf_err(nc_status)       
c----------------------------------------------------------------------
c     Write restart file
c----------------------------------------------------------------------
      if(iyr.ge.2000) write(ay,'(i2.2)') iyr-2000
      if(iyr.lt.2000) write(ay,'(i2.2)') iyr-1900
      write(am,'(i2.2)') imm
      write(ad,'(i2.2)') idd
      write(ah,'(i2.2)') istart/100
      write(a3,'(i3.3)') jtime + ihrestart
      open(98,file=TRIM(experiment_folder)//
     &        "/xp_files/"//ay//am//ad//ah//'_'//a3//'.rso',
     &        form='unformatted')
      write(98) jtime+ihrestart,npcl,nspill,pcevp,pcsrf,pcdsp,pccst,
     &          deno,viso

      do i=1,npcl
        vcst(i)=0.d0
        if(is(i).lt.0) vcst(i) = seep(-int(is(i)))
        alon = glon(px(i))
        alat = glat(py(i))
        write(98) is(i),ib(i),c1p(i),c2p(i),alon,alat,pz(i),vcst(i) 
      enddo

      do i=1,nspill
        write(98) den(i),vis(i),visem(i),tre(i),c1ms(i),c2ms(i),
     &    atn(i),atk(i),ato(i),ttk(i),tto(i),
     &    vtn(i),vtk(i),vto(i),xcl(i),xcs(i),
     &    vtne(i),vtke(i),vte(i),vtnd(i),vtkd(i),vtd(i),
     &    ftk(i),ftn(i),fw(i),pcte(i),pctd(i) 
      enddo

      do ns=1,nseg
        if(seep(ns).gt.0.d0) then
          xx1 = glon(seg(ns,1)) 
          yy1 = glat(seg(ns,2)) 
          write(98) ns,xx1,yy1,seep(ns)
        endif   
      enddo

      close(98)
      close(90)
      end

c **********************************************************************
c     OIL FATE routines
c **********************************************************************
c----------------------------------------------------------------------
c Constructs coastal segments from regional map 
c----------------------------------------------------------------------
      subroutine setcst(preproc_folder)
      implicit real*8(a-h,o-z)
      parameter(ndim=1000, nss=200000)
      integer idpoint1,idpoint2
      integer, dimension(nss,2) :: idpointbd
      integer, dimension(nss) :: ibd
      dimension bd1(nss,4),
     &          cstlat(ndim),cstlon(ndim),icst(ndim)
      character empty*80,preproc_folder*400
      logical ex
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      open(51,file=TRIM(preproc_folder)//'/bnc_files/dtm.map',
     &        status='old')
      open(52,file=TRIM(preproc_folder)//'/bnc_files/dtmcst1.d')
      read(51,*) ncontours
      idpoint1 = 1
      idpoint2 = 1
      k=1
      do ni=1,ncontours
        read(51,*) isle
        read(51,*) x1,y1
        do i=2,isle
          read(51,*) x2,y2
          if( (x1.ge.along1.and.x1.le.along2.and.
     &         y1.ge.alatg1.and.y1.le.alatg2).or. 
     &        (x2.ge.along1.and.x2.le.along2.and.
     &         y2.ge.alatg1.and.y2.le.alatg2) ) then
            bd1(k,1) = x1 
            bd1(k,2) = y1 
            bd1(k,3) = x2 
            bd1(k,4) = y2
            idpointbd(k,1) = idpoint1 
            idpointbd(k,2) = idpoint2
            k = k + 1
          endif           
          x1 = x2
          y1 = y2
          idpoint1 = idpoint2
        enddo 
      enddo
      nsegt = k-1
      write(90,*) 'No of coastal segments = ',nseg
      close(51)      
      do ns=1,nsegt
        ibd(ns) = 1
      enddo
c     read beach types
      ncstty = 0
c      output results
      write(52,'(''no. of segments = '',i6)') nsegt
      do ns=1,nsegt
        write(52,100) (bd1(ns,j),j=1,4),ibd(ns),(idpointbd(ns,j),j=1,2)
      enddo
  100 format(4f12.7,i6,2i10)
      rewind(52)
      return
      end

c----------------------------------------------------------------------
c     Reads coastal segment data
c     selects those coastal segments that are close enough to spill
c----------------------------------------------------------------------
      subroutine coast(dt,seg,sfrac,prel,nseg,alngt,
     &       api,apicoeff,preproc_folder,itype,bseg1_pid, bseg2_pid)  
      implicit real*8(a-h,o-z)
      parameter(nss=200000)
      integer mm,nm
      common /size/ mm,nm
      save ind,iused,ibd,bd1,nsegt,dkm
      integer*2 iused(nss)
      integer, dimension(nss) :: ibd
      character preproc_folder*400
      integer, dimension(nss) :: bseg1, bseg2,
     &                    bseg1_pid, bseg2_pid   
      dimension bd1(nss,4),dkm(nss),itype(mm,nm),
     &          seg(nss,4),prel(nss),sfrac(nss),alngt(nss)
      common /blk3/ delx,dely,pi,degrad
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      common /blk5/ mzb,mzf,nzb,nzf
      data ind /0/
      xgrid(alon)=(alon - along1) / dlong + 1.d0
      ygrid(alat)=(alat - alatg1) / dlatg + 1.d0
      glon(x)=along1+(x-1.d0)*dlong
      glat(y)=alatg1+(y-1.d0)*dlatg
c     coeff for reduction of coastal retention rate for heavy oils
      cseep = 1.d0
      if(api.lt.30.) cseep = 1. + (30.-api) * apicoeff
c   ....convert the boundary segments to grid coords....
c   ....ibd(n) = shore type of segment n (e.g. sand, rock, mangrove etc.)
c   ....dkm(n) = length of segment n in km.
      if(ind.eq.0) then
        call setcst(preproc_folder)
        read(52,'(18x,i6)') nsegt
        do ns=1,nsegt
          read(52,100) (bd1(ns,j),j=1,4),ibd(ns),bseg1(ns),bseg2(ns)
          do j=1,3,2
            bd1(ns,j)   = xgrid(bd1(ns,j))
            bd1(ns,j+1) = ygrid(bd1(ns,j+1))
          enddo      
          dx = ( bd1(ns,3) - bd1(ns,1) ) * delx
          dy = ( bd1(ns,4) - bd1(ns,2) ) * dely
          dkm(ns) = dsqrt(dx*dx + dy*dy) / 1000.d0
        enddo
        close(52)
      endif
      ind=1
  100 format(4f12.7,i6,2i10)
c   ....xb - xf, yb - yf: expected limits of spill....
      xb=mzb
      xf=mzf
      yb=nzb
      yf=nzf
c     select boundary segments that are close enough to spill site....
c     prel(j) = prob of release from coastal segment j in time step dt
c     sfrac(j) = fraction absorbed onto coast in time step dt
c     alngt(j) = length of segment j in km
      j=nseg
      do 20 ns=1,nsegt
        if(iused(ns).eq.1) go to 20
        x1=bd1(ns,1)
        y1=bd1(ns,2)
        x2=bd1(ns,3)
        y2=bd1(ns,4)
        if((x1.ge.xb.and.x1.le.xf.and.y1.gt.yb.and.y1.le.yf).or.
     &     (x2.ge.xb.and.x2.le.xf.and.y2.gt.yb.and.y2.le.yf)) then
          j=j+1
          if(j.gt.100000) then
            write(6,*) 'Too many coastal segments are impacted:' 
            write(6,*) '           reduce length of computation'
            stop
          endif
          iused(ns)=1  
          seg(j,1)=x1
          seg(j,2)=y1
          seg(j,3)=x2
          seg(j,4)=y2
          ib=ibd(ns)
          bseg1_pid(j)=bseg1(ns)
          bseg2_pid(j)=bseg2(ns)
          tseep=hlseep(ib) * cseep
          twash=hlwash(ib)
          if(twash.eq.0.) then
            prel(j)=1.0d0
            sfrac(j)=0.0d0
          else   
            prel(j)  = 1.0d0 - 0.5d0**(dt/twash)
            sfrac(j) = 1.0d0 - 0.5d0**(dt/tseep)
          endif
          alngt(j)=dkm(ns)
        endif
   20 continue
      nseg=j  
  199 format(i3,4f10.3,i6,2f10.5)     
      return
      end             

c----------------------------------------------------------------------
c   ....half-life for seepage into the beach; beach type = ib
c   ....coastal types ib:                   
c     1   sand beach                                                            
c     2   sand and gravel beach                                                 
c     3   cobble beach                                                          
c     4   rocky shore                                                           
c     5   seawall; concrete wharf etc                                                    
c     6   exposed headland                                                      
c     7   sheltered sand or gravel beach                                        
c     8   sheltered rocky shore                                                 
c     9   sheltered marsh or mud flats                                          
c----------------------------------------------------------------------
      function hlseep(ib)
      implicit real*8(a-h,o-z)
      if(ib.eq.1) hlseep= 24.d0 
      if(ib.eq.2) hlseep= 36.d0
      if(ib.eq.3) hlseep= 48.d0
      if(ib.eq.4) hlseep= 96.d0 
      if(ib.eq.5) hlseep= 96.d0
      if(ib.eq.6) hlseep= 96.d0 
      if(ib.eq.7) hlseep= 24.d0
      if(ib.eq.8) hlseep= 96.d0
      if(ib.eq.9) hlseep= 24.d0
      return
      end

c----------------------------------------------------------------------
c   ....half-life for washing off the beach; beach type = ib
c----------------------------------------------------------------------
      function hlwash(ib)
      implicit real*8(a-h,o-z)
      if(ib.eq.1) hlwash= 24.d0
      if(ib.eq.2) hlwash= 24.d0
      if(ib.eq.3) hlwash= 24.d0
      if(ib.eq.4) hlwash= 18.d0
      if(ib.eq.5) hlwash=  0.d0
      if(ib.eq.6) hlwash=  1.d0 
      if(ib.eq.7) hlwash=120.d0
      if(ib.eq.8) hlwash=120.d0
      if(ib.eq.9) hlwash=120.d0
      return
      end      

c----------------------------------------------------------------------
c     Computes evaporation probability (eprob) and
c           dispersion probability (dprob) and emulsification rate
c     Program is modified from Mackay, Paterson and Trudel, 'Mathematical 
c         Model of Oil Spill Behaviour', Dec 1980
c----------------------------------------------------------------------
      subroutine ed(dtim,vmspl,den,vis,
     &              visem,ttk,ttn,tto,
     &              atk,atn,ato,vtk,vtn,vto,ftk,ftn,ft,fw,
     &              vtke,vtne,vte,vtkd,vtnd,vtd,
     &              xcl,xcs,pcte,pctd)
      implicit real*8(a-h,o-z)
      common /evap/ ce1,ce,vappr,fmaxe
      common /disp/ cd1,cd3,cd4,cd5,vl,vs1,um,stk,stn,fmaxd
      common /emul/ cm1,cm2,cm3
      common /sprd/ cs1,cs2,cs3
      common /phys/ deno,denk,cdt,viso,visk,cvt,tvk0,denw,tk,tk0,wvel
      common /weath/ ned
c----------------------------------------------------------------------
      ! EVAPORATION THICK slick
      dvtke = 0.d0
      dftk = 0.d0
      if(vtk.gt.0.d0) then
        detk = ce1*atk*dtim*(1.d0-ftk)/vtk
        poil = tk*8.2e-05/2.0e-04
        vpoil = vappr*exp(-ce*ftk)/poil
        dftk = detk*vpoil
        ! ************************************
        ! This is done for numerical reasons,
        ! to make the evaporation curve more
        ! asymptotic, especially for heavy oils.
        ! It is not present in the
        ! reference equations.
        threshold = 5.d0
        dftkmax = (fmaxe-ftk)/(ned*threshold)
        if(dftk.gt.dftkmax) dftk = dftkmax
        ! ************************************
        dvtke = vtk*dftk/(1.d0-ftk)
      endif
      ftk = ftk+dftk
      if(ftk.gt.fmaxe) ftk = fmaxe
      vtke = vtke+dvtke
      ! EVAPORATION THIN slick
      dvtne = 0.d0
      if(vtn.gt.0.d0) then
        if(ftn.le.fmaxe) dvtne = vtn*(fmaxe-ftn)/(1.d0-ftn)
      endif
      ftn = fmaxe
      vtne = vtne+dvtne
      ! TOTAL evaporation
      vte = vtke+vtne
      pcte = 100.d0*vte/vmspl
      if(pcte.ge.fmaxe*100.d0) pcte = fmaxe*100.d0
c----------------------------------------------------------------------
      ! EMULSIFICATION
      dfw = cm2*(wvel+1.d0)**2*(1.d0-cm3*fw)*dtim
      fw = fw+dfw
      if(fw.gt.1.d0/cm3) fw = 1.d0/cm3
      if (ttk.gt.0.d0) then
        vis = viso * exp(visk * ftk)
        ! density of oil, NOT used in weathering
        den = denw*fw+deno*(1.d0-fw)*(1.d0+denk*ftk)
      endif
      visem = vis*exp(2.5d0*fw/(1.d0-cm1*fw))
c----------------------------------------------------------------------
c     DISPERSION
      pctd_max = fmaxd*100.d0
      if(pctd.ge.pctd_max) go to 30
      dfd = cd3*(wvel+1.0)**2
c----------------------------------------------------------------------
c     dispersion in thick slick
c     xs = volume of small drops dispersed
      dvtkd = 0.d0
      if(vtk.gt.0.d0.and.
     &   ttk.gt.0.d0) then
        xcs_previous = xcs
        fstk = 1.d0/(1.d0+cd4*((visem/10.d0)**(0.5d0))*
     &         (ttk/0.001d0)*(stk/24.d0))
        cs = 2.d0*dfd*ttk*fstk/(vs1+cd1)
        xcs = cs*um*atk
        dxs = xcs - xcs_previous
        dvtkd = 0.5d0*(cd1-vs1)*cs*atk*dtim + dxs
      endif
      vtkd = vtkd+dvtkd ! update thick slick dispersed volume
c----------------------------------------------------------------------
c     dispersion in thin slick
      dvtnd = 0.d0
      if (vtn.gt.0.d0) then
        fstn = 1.d0/(1.d0+cd5*stn/24.d0)
        dvtnd = dfd*ttn*atn*fstn*dtim
      endif
      vtnd = vtnd+dvtnd
c     TOTAL dispersion
      vtd = vtkd+vtnd
      pctd = vtd*100.d0/vmspl
   30 continue
      if(pctd.ge.pctd_max) pctd = pctd_max
c----------------------------------------------------------------------
c     spreading: use Fay model for areas of both thick and thin slicks
c----------------------------------------------------------------------
      dvtns = 0.d0
      datks = 0.d0
      if(vtk.gt.0.) then
        datns=cs1*(atn**0.333d0)*exp(-cs3/(ttk+0.00001d0))*dtim
        dvtns = ttn*datns
        dvtks = -dvtns
        datks = dvtks/ttk + cs2*atk**0.333d0*ttk**1.333d0*dtim
      endif
c----------------------------------------------------------------------
c     Update structural variables
c     transfer thick slick to thin slick if ttk <= ttn
c----------------------------------------------------------------------
      atk = atk+datks
      vtk = vtk-dvtke-dvtkd-dvtns
      vtn = vtn-dvtne-dvtnd+dvtns
      ttk = vtk/atk
      if(ttk.le.ttn) then
        vtn = vtn+vtk
        vtk = 0.d0
        atk = 0.d0
        ttk = 0.d0
        xcl = 0.d0
        xcs = 0.d0
      endif
      atn = vtn/ttn
      vto = vtn+vtk
      ato = atk+atn
      tto = vto/ato
c----------------------------------------------------------------------
c     calculate new evaporative fraction of thin slick after spreading
c----------------------------------------------------------------------
      ftn = fmaxe-dvtns*(fmaxe-ftk)/vtn

      return
      end

c **********************************************************************
c     DIFFUSIVITY routines
c **********************************************************************
c----------------------------------------------------------------------
c     computes the mean diffusion step at depth 'dep'
c----------------------------------------------------------------------
      function vertd(dep,vertd1,vertd2,thermocl)
      implicit real*8(a-h,o-z)
      if(dep.lt.thermocl) then
        vertd=vertd1
      else
        vertd=vertd2
      endif
      return
      end             

c----------------------------------------------------------------------
c     compute horizontal diffusity from Smagorinsky model
c----------------------------------------------------------------------
      subroutine smag(x0,y0,us,vs,horizk,itype)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      dimension itype(mm,nm),us(mm,nm),vs(mm,nm)
      common /blk3/ delx,dely,pi,degrad
      data c /0.1d0/
      m=int(x0+0.5d0)
      n=int(y0+0.5d0)

      if(itype(m,n).eq.0) then
        ux=0.d0
        vx=0.d0
      else if(itype(m+1,n).ne.0.and.itype(m-1,n).ne.0) then
        ux=(us(m+1,n)-us(m-1,n))/(2.d0*delx)
        vx=(vs(m+1,n)-vs(m-1,n))/(2.d0*delx)
      else if(itype(m+1,n).ne.0.and.itype(m-1,n).eq.0) then
        ux=(us(m+1,n)-us(m,n))/(delx)
        vx=(vs(m+1,n)-vs(m,n))/(delx)
      else if(itype(m+1,n).eq.0.and.itype(m-1,n).ne.0) then
        ux=(us(m,n)-us(m-1,n))/(delx)
        vx=(vs(m,n)-vs(m-1,n))/(delx)
      else 
        ux=0.d0 
        vx=0.d0
      endif
      if(itype(m,n).eq.0) then
        uy=0.d0
        uy=0.d0
      else if(itype(m,n+1).ne.0.and.itype(m,n-1).ne.0) then
        uy=(us(m,n+1)-us(m,n-1))/(2.d0*dely)
        vy=(vs(m,n+1)-vs(m,n-1))/(2.d0*dely)
      else if(itype(m,n+1).ne.0.and.itype(m,n-1).eq.0) then
        uy=(us(m,n+1)-us(m,n))/(dely)
        vy=(vs(m,n+1)-vs(m,n))/(dely)
      else if(itype(m,n+1).eq.0.and.itype(m,n-1).ne.0) then
        uy=(us(m,n)-us(m,n-1))/(dely)
        vy=(vs(m,n)-vs(m,n-1))/(dely)
      else 
        uy=0.d0
        vy=0.d0
      endif
      fac=ux*ux+0.5d0*(vx+uy)*(vx+uy)+vy*vy
      horizk=c*delx*dely*dsqrt(fac)
      if(horizk.lt.0.5d0) horizk=0.5d0
      if(horizk.gt.15.d0) horizk=15.d0
      return
      end
    
c **********************************************************************
c     WIND routines
c **********************************************************************
c----------------------------------------------------------------------
c     constructs wind velocity field in case of forecast wind
c     (winx,winy) = forecast wind velocity at grid point (m,n)
c     (dwinx,dwiny) = increment per time step
c     (modified by M. De Dominicis)
c----------------------------------------------------------------------
      subroutine ski_ecmwf_wind(xavg,yavg,nst,time,delt,nwfcst,
     &                  wfcstfn,iwfcstfn,wfcsttim,winx,
     &                  winy,wvel,wdir,dwinx,dwiny,
     &                  preproc_folder, itype)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm 
      real*8,dimension(mm,nm) :: winx,winy,dwinx,dwiny
      dimension :: itype(mm,nm)
      save isub,readdata,ifile,iwindrec
      dimension iwfcstfn(30),wfcsttim(30,24),temp(24)
      common /spill/  idd,imm,iyr,ispill,tstart,tcomp,x0,y0
      common /blk3/ delx,dely,pi,degrad
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      character dirr*400,wfcstfn(30)*6,fn*14,preproc_folder*400
      logical readdata
      glon(x)=along1+(x-1.d0)*dlong
      glat(y)=alatg1+(y-1.d0)*dlatg
      data isub /1/
      if(isub.eq.0) return         ! end of available data files
      m1=int(xavg+0.5d0)           ! centre of slick
      n1=int(yavg+0.5d0)
      slon=glon(dfloat(m1))
      slat=glat(dfloat(n1))
      nrecs=24        
      dtwind = 24.d0 / dfloat(nrecs)             
      timew = time - delt / 2.d0             ! timew = time at middle of step
      frac = dtwind / delt          !no of steps between successive wind records
      frac2 = frac + 0.5d0 
      dirr=TRIM(preproc_folder)//'/met_files/' !str
c     On 1st step set times of forecast records: wfcsttim(i,k) = time of 
c     the kth record in ith file measured from 0 hrs on spill date
c      Then set the initial wind forecast file and record
c     fac = no of timesteps from 1st wind record to middle of 1st step
      if(nst.eq.1) then
        write(90,*) ' '
        write(90,*) 'Wind forecast files and times of wind records:'
        do i=1,nwfcst
          read(wfcstfn(i)(1:2),'(i2)') iy
          read(wfcstfn(i)(3:4),'(i2)') im
          read(wfcstfn(i)(5:6),'(i2)') id
          ndaym1 = jdiff(idd,imm,iyr,id,im,iy+2000)
          do k = 1,nrecs
            wfcsttim(i,k) = ndaym1 * 24.d0 + (k-1) * dtwind
            temp(nrecs + 1 - k) = 24.d0 - wfcsttim(i,k)
          enddo
          write(90,'(a14,2x,24f6.0)') wfcstfn(i),
     &        (wfcsttim(i,k),k=1,nrecs)
        enddo
        readdata=.true.
        ifile=1
        do k=1,nrecs
          if(timew.ge.wfcsttim(ifile,k)) iwindrec=k
        enddo
        fac=(timew - wfcsttim(ifile,iwindrec))/delt
      endif
c  read a fresh pair of wind records
      if(timew.ge.wfcsttim(ifile,iwindrec)) then
        readdata=.true.
        write(90,*) timew,wfcsttim(ifile,iwindrec)
      endif
      if(readdata) then
        readdata=.false.
        fn="erai"//wfcstfn(ifile)//".eri"
        iw = iwindrec
        write(6,*) 'Reading wind from file '//fn//', record ',iw
        write(90,*) 'Reading wind from file '//fn//', record ',iw
        call readwind(dirr,fn,iw,nrecs,winx,winy)
        if(iwindrec.eq.nrecs) then
          iwindrec=1
          ifile=ifile+1
        else
          iwindrec=iwindrec+1
        endif
        if(iwfcstfn(ifile).eq.0) then
          isub=0
          return
        endif
        fn="erai"//wfcstfn(ifile)//".eri"
        iw = iwindrec
        write(6,*) 'Reading wind from file '//fn//', record ',iw
        write(90,*) 'Reading wind from file '//fn//', record ',iw  
        call readwind(dirr,fn,iw,nrecs,dwinx,dwiny)
c  calculate the increments for each spill re-computation
c  on initial step compute the values for the time of the spill
c     After 1st step, winx, winy are at half timestep behind 'time'
        do m=1,mm
          do n=1,nm
            if(nst.eq.1) then
              dwinx(m,n) = (dwinx(m,n)-winx(m,n)) / frac
              dwiny(m,n) = (dwiny(m,n)-winy(m,n)) / frac
              winx(m,n) = winx(m,n) + dwinx(m,n) * fac
              winy(m,n) = winy(m,n) + dwiny(m,n) * fac
            elseif(nst.gt.1) then
              dwinx(m,n) = (dwinx(m,n)-winx(m,n)) / frac2
              dwiny(m,n) = (dwiny(m,n)-winy(m,n)) / frac2
            endif
          enddo
        enddo
      else
c  increment the values for each re-computation of the spill
        do m=1,mm
          do n=1,nm
            winx(m,n)  = winx(m,n)  + dwinx(m,n)
            winy(m,n)  = winy(m,n)  + dwiny(m,n)
          enddo
        enddo
      endif
c  compute wind speed and direction at centre of spill
      wxx=winx(m1,n1)
      wyy=winy(m1,n1)
      wvel=dsqrt(wxx*wxx+wyy*wyy)
      wdir=0.d0
      if(wxx.eq.0.d0) then
        wdir=0.d0
        if(wyy.gt.0.d0) wdir=180.d0
        if(wyy.le.0.d0) wdir=0.d0
      else
        wdir=datan(wyy/wxx) * degrad
        if(wxx.lt.0.d0) wdir=wdir+180.d0
        wdir=270.d0-wdir
      endif
      return
      end

c----------------------------------------------------------------------
c     read wind forecast data and interpolate to Medslik grid
c     optional interpolation also over land points
c----------------------------------------------------------------------
      subroutine readwind(dirr,fn,iwindrec,nrecs,wfx,wfy)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      dimension wfx(mm,nm),wfy(mm,nm),wfx2(mm,nm),wfy2(mm,nm),
     &          itype(mm,nm),wx1(24),wy1(24)
      character dirr*400,fn*14,dummy*80,acall*2
      common /blk3/ delx,dely,pi,degrad
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      open(72,file=TRIM(dirr)//fn)
      read(72,*) dummy
      read(72,*) dummy
      read(72,*) alon1,alon2,alat1,alat2,mmaxs,nmaxs
      read(72,*) nlines
      read(72,*) dummy
      dlon=(alon2-alon1)/dfloat(mmaxs-1)
      dlat=(alat2-alat1)/dfloat(nmaxs-1)
      do i=1,nlines
        read(72,*) alat,alon,(wx1(k),wy1(k),k=1,nrecs)
        m=int((alon-alon1)/dlon+1.01d0)
        n=int((alat-alat1)/dlat+1.01d0)
        wfx(m,n)=wx1(iwindrec)
        wfy(m,n)=wy1(iwindrec)  
      enddo
      close(72)
c     now interpolate data from skiron grid onto medslik grid
c     comment out the line (*) to include interpolation to land points
      do m=1,mm
        do n=1,nm
          wfx2(m,n)=0.d0
          wfy2(m,n)=0.d0
          x = along1 + dfloat(m-1) * dlong
          y = alatg1 + dfloat(n-1) * dlatg
          xdata = (x - alon1) / dlon + 1.d0
          ydata = (y - alat1) / dlat + 1.d0
          mdata=int(xdata)
          ndata=int(ydata)
          if(.not.(mdata.lt.1.or.ndata.lt.1
     &        .or.mdata.gt.mmaxs.or.ndata.gt.nmaxs)) then
          wfx2(m,n)=(wfx(mdata,ndata)*(dfloat(mdata+1)-xdata)+
     &                wfx(mdata+1,ndata)*(xdata-dfloat(mdata)))
     &                                       * (dfloat(ndata+1)-ydata)
     &           +(wfx(mdata,ndata+1)*(dfloat(mdata+1)-xdata)+
     &             wfx(mdata+1,ndata+1)*(xdata-dfloat(mdata))) 
     &                                       * (ydata-dfloat(ndata))
          wfy2(m,n)=(wfy(mdata,ndata)*(dfloat(mdata+1)-xdata)+
     &                wfy(mdata+1,ndata)*(xdata-dfloat(mdata)))
     &                                       * (dfloat(ndata+1)-ydata)

     &           +(wfy(mdata,ndata+1)*(dfloat(mdata+1)-xdata)+
     &             wfy(mdata+1,ndata+1)*(xdata-dfloat(mdata))) 
     &                                       * (ydata-dfloat(ndata))
          endif
        enddo
      enddo  
      do m=1,mm
        do n=1,nm
          wfx(m,n)=wfx2(m,n)
          wfy(m,n)=wfy2(m,n)
        enddo
      enddo
      return
      end

c **********************************************************************
c     WATER CURRENT routines
c **********************************************************************
c----------------------------------------------------------------------
c     constructs current field in case of HOURLY forecast data
c     (MFS, Sicily, Adriatic and Tyrrhenian models)
c     time = current time in hours after 0 hrs on date of spill
c     (added by M. De Dominicis)
c----------------------------------------------------------------------
      subroutine fcstcur_1hr(nst,time,delt,nfcst,
     &      fcstfn,ifcstfn,fcsttim,
     &      fcstcurdir,sst,wx,
     &      wy,us,vs,u10,v10,
     &      u30,v30,u120,v120,dsst,
     &      dwx,dwy,dus,dvs,
     &      du10,dv10,du30,dv30,
     &      du120,dv120,itype)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm   
      save ifile,isub
      dimension ifcstfn(720),fcsttim(720),itype(mm,nm)
      real*8,dimension(mm,nm) :: us,vs,dus,dvs,
     &      sst,wx,wy,
     &      u10,v10,u30,v30,
     &      u120,v120,dsst,dwx,dwy,
     &      du10,dv10,
     &      du30,dv30,du120,dv120 
      character fcstcurdir*400, fcstfn(720)*16, fn*16, a2*2

      common /spill/  idd,imm,iyr,ispill,tstart,tcomp,x0,y0
      common /blk3/ delx,dely,pi,degrad    
      data ifile /1/,isub /1/
      if(isub.eq.0) return         ! end of available data files
c     on 1st step set the times of the forecast files (hrs after 0 hrs 
c      on spill date) then open the initial forecast file and read initial data
c     On 1st step, fac is used later to interpolate to mid-step
c     For hindcast all times are backwards from 2400 hrs on start date
      if(nst.eq.1) then
        write(90,*) ' '
        write(90,*) 'Frcst files and file times from 0 hrs on spill day'



        nfcst=nfcst*24
        do i=1,nfcst
          a2=fcstfn(i)(5:6)

          a2=fcstfn(i)(7:8)
          read(a2,'(i2)') im
          a2=fcstfn(i)(9:10)
          read(a2,'(i2)') id
          a2=fcstfn(i)(11:12)
          read(a2,'(i2)') ih

          if(dfloat(ih).eq.int(tstart).and.idd.eq.id) then
            ifile=i

          endif
        enddo


        do i=ifile,nfcst

          a2=fcstfn(i)(5:6)
          read(a2,'(i2)') iy
          a2=fcstfn(i)(7:8)
          read(a2,'(i2)') im
          a2=fcstfn(i)(9:10)
          read(a2,'(i2)') id
          a2=fcstfn(i)(11:12)
          read(a2,'(i2)') ih
          nday = jdiff(idd,imm,iyr,id,im,iy+2000)


          fcsttim(i)=nday*24.d0 + dfloat(ih)-0.5 !average

          write(90,*) i,'   ',fcstfn(i),fcsttim(i)

        enddo
        write(90,*) ' '

        fn=fcstfn(ifile)
        write(6,*) 'Reading forecast currents from file ',fn
        write(90,*) 'Forecast current directory = ',fcstcurdir
        write(90,*) 'Reading forecast currents from file ',fn
        write(90,*) ' '    
         call readfcst_1hr(fcstcurdir,fn,sst,us,vs,u10,v10,
     &                           u30,v30,u120,v120)
        fac=(time - 0.5d0*delt - fcsttim(ifile)) / delt
      endif  
c  open the next forecast file and read new data (on 1st step file #2)
c     time has already been advanced to the end of the current step
      if(time - 0.5d0*delt .ge. fcsttim(ifile)) then
        ifile=ifile+1
        if(ifcstfn(ifile).eq.0) then      ! past the last available file
          isub=0                      
          write(6,*) 'WARNING: Not enough forecast data is available.' 
          write(6,*) 'Forecast data will be kept constant from now on' 
          go to 22
        endif
        fn=fcstfn(ifile)
        write(6,*) 'Reading forecast currents from file ',fn
        write(90,*) 'Reading forecast currents from file ',fn
        write(90,*) ' '
        call readfcst_1hr(fcstcurdir,fn,dsst,dus,dvs,du10,dv10,
     &                              du30,dv30,du120,dv120)
        dtfcst=fcsttim(ifile)-fcsttim(ifile-1)
        frac = dtfcst / delt          !no of steps between successive fcsts
        frac2 = frac + 0.5d0 
c  calculate the increments for each spill re-computation
c     remember after 1st step, sst, wx, etc are at mid-step
c       delt/2 behind the time of the newly-read dsst, dwx, etc
c  initial step compute values for mid-step =  spilltime + delt/2
        do m=1,mm
          do n=1,nm
            if(nst.eq.1) then
              dsst(m,n)  = (dsst(m,n)  -sst(m,n))  / frac
              dus(m,n)   = (dus(m,n)   -us(m,n))   / frac
              dvs(m,n)   = (dvs(m,n)   -vs(m,n))   / frac
              du10(m,n)  = (du10(m,n)  -u10(m,n))  / frac
              dv10(m,n)  = (dv10(m,n)  -v10(m,n))  / frac
              du30(m,n)  = (du30(m,n)  -u30(m,n))  / frac
              dv30(m,n)  = (dv30(m,n)  -v30(m,n))  / frac
              du120(m,n) = (du120(m,n) -u120(m,n)) / frac
              dv120(m,n) = (dv120(m,n) -v120(m,n)) / frac

              sst(m,n)  = sst(m,n)  + dsst(m,n)  * fac
              us(m,n)   = us(m,n)   + dus(m,n)   * fac
              vs(m,n)   = vs(m,n)   + dvs(m,n)   * fac
              u10(m,n)  = u10(m,n)  + du10(m,n)  * fac
              v10(m,n)  = v10(m,n)  + dv10(m,n)  * fac
              u30(m,n)  = u30(m,n)  + du30(m,n)  * fac
              v30(m,n)  = v30(m,n)  + dv30(m,n)  * fac
              u120(m,n) = u120(m,n) + du120(m,n) * fac
              v120(m,n) = v120(m,n) + dv120(m,n) * fac

            elseif(nst.gt.1) then

              dsst(m,n)  = (dsst(m,n)  -sst(m,n))  / frac2
              dus(m,n)   = (dus(m,n)   -us(m,n))   / frac2
              dvs(m,n)   = (dvs(m,n)   -vs(m,n))   / frac2
              du10(m,n)  = (du10(m,n)  -u10(m,n))  / frac2
              dv10(m,n)  = (dv10(m,n)  -v10(m,n))  / frac2
              du30(m,n)  = (du30(m,n)  -u30(m,n))  / frac2
              dv30(m,n)  = (dv30(m,n)  -v30(m,n))  / frac2
              du120(m,n) = (du120(m,n) -u120(m,n)) / frac2
              dv120(m,n) = (dv120(m,n) -v120(m,n)) / frac2

            endif

          enddo
        enddo
      endif  
c  increment the values for each re-computation of the spill except first
      if(nst.ne.1) then
        do m=1,mm
          do n=1,nm
            sst(m,n)  = sst(m,n)  + dsst(m,n)
            us(m,n)   = us(m,n)   + dus(m,n)
            vs(m,n)   = vs(m,n)   + dvs(m,n)
            u10(m,n)  = u10(m,n)  + du10(m,n)
            v10(m,n)  = v10(m,n)  + dv10(m,n)
            u30(m,n)  = u30(m,n)  + du30(m,n)
            v30(m,n)  = v30(m,n)  + dv30(m,n)
            u120(m,n) = u120(m,n) + du120(m,n)
            v120(m,n) = v120(m,n) + dv120(m,n)
          enddo
        enddo
      endif
   22 continue
      return
      end

c----------------------------------------------------------------------
c     read forecast hourly current data and interpolate to Medslik grid
c     (added by M. De Dominicis)
c----------------------------------------------------------------------
      subroutine readfcst_1hr(fcstcurdir,fn,sst,us,vs,u10,v10,
     &                     u30,v30,u120,v120)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      real*8,dimension(mm,nm) :: sst,
     &          us,vs,u10,v10,
     &          u30,v30,u120,v120   
      character fcstcurdir*400,fn*16,dummy*80
      common /blk4/ along1,alatg1,along2,alatg2,dlong,dlatg
      open(71,file=TRIM(fcstcurdir)//fn)
      read(71,*) dummy
      read(71,*) dummy
      read(71,*) alon1,alon2,alat1,alat2,mmaxc,nmaxc
      if(mmaxc.ne.mm.or.nmaxc.ne.nm) then
        print *, "ERROR: Currents and bathymetry
     &             have different dimensions"
        print *, "Bathymetry has shape ", mm, nm
        print *, "while Current field has shape ", mmaxc, nmaxc
        stop
      endif
      read(71,*) ndata
      read(71,*) dummy
      do m=1,mm
        do n=1,nm
          sst(m,n)=0.d0
          us(m,n)=0.d0
          vs(m,n)=0.d0
          u10(m,n)=0.d0
          v10(m,n)=0.d0
          u30(m,n)=0.d0
          v30(m,n)=0.d0
          u120(m,n)=0.d0
          v120(m,n)=0.d0
        enddo
      enddo
      do i=1,ndata
        read(71,*) alat,alon,st,u,v,u1,v1,u3,v3,u4,v4
        m=int((alon-along1)/dlong+1.1d0)
        n=int((alat-alatg1)/dlatg+1.1d0)        
        if(m.gt.mm.or.n.gt.nm.or.m.lt.1.or.n.lt.1) go to 1 
        sst(m,n)=st
        us(m,n)=u
        vs(m,n)=v
        u10(m,n)=u1
        v10(m,n)=v1
        u30(m,n)=u3
        v30(m,n)=v3
        u120(m,n)=u4
        v120(m,n)=v4
    1   continue
       enddo
      close(71)
      return
      end

c **********************************************************************
c     UTILITY routines
c **********************************************************************

c**********************************************************************
c     compute Sea Over Land
c----------------------------------------------------------------------
      subroutine sea_over_land(datasource)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      dimension datasource(mm,nm), carpet(mm,nm)
      do iter=1,5
        carpet(:,:)=datasource(:,:)
        do j=1,nm
          do i=1,mm
            if(datasource(i,j).eq.0) then  ! if we are on land
              datan=0.
              jcn=0
              im1=i-1
              ip1=i+1
              jm1=j-1
              jp1=j+1
              do jj=jm1,jp1
                do ii=im1,ip1
                  if(datasource(ii,jj).ne.0) then
                    datan=datan+datasource(ii,jj) ! sum of the values of sea points
                    jcn=jcn+1  ! number of sea points 
                  endif
                enddo 
              enddo 
              if(jcn.ge.2) then ! if we have at least one sea neighborhood 
                carpet(i,j)=datan/float(jcn)
              endif
            endif 
          enddo 
        enddo
        datasource(:,:)=carpet(:,:)
      enddo
      return
      end      

c**********************************************************************
c     interpolates values of an array q
c     from grid values to particle position (x,y)
c----------------------------------------------------------------------
      subroutine intrpl0(x,y,q,qint)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      dimension q(mm,nm)
      m=int(x)
      n=int(y)
      q1=q(m,n)
      q2=q(m+1,n)
      q3=q(m+1,n+1)
      q4=q(m,n+1)
      qint=(q1*(m+1-x)+q2*(x-m))*(n+1-y)+
     &         (q4*(m+1-x)+q3*(x-m))*(y-n)
      return
      end

c**********************************************************************
c     interpolates values of an array q from grid 
c     values to a point (x,y) taking account of land/water mask
c----------------------------------------------------------------------
      subroutine intrpl(x,y,q,itype,qint)
      implicit real*8(a-h,o-z)
      integer mm,nm
      common /size/ mm,nm
      parameter(ntm=2000,npc=100000,nss=200000)
      dimension q(mm,nm),itype(mm,nm)
      m=int(x)
      n=int(y)
      ! check if land or sea
      it1=itype(m,n)
      it2=itype(m+1,n)
      it3=itype(m+1,n+1)
      it4=itype(m,n+1)
      q1=q(m,n)
      q2=q(m+1,n)
      q3=q(m+1,n+1)
      q4=q(m,n+1)
      ! SEA OVER LAND PROCEDURE - constant gradient method
      if(it1.eq.0.and.it2.ne.0.and.it3.ne.0.and.it4.ne.0) then
        q1=q2+q4-q3
      else if(it1.ne.0.and.it2.eq.0.and.it3.ne.0.and.it4.ne.0) then
        q2=q1+q3-q4
      else if(it1.ne.0.and.it2.ne.0.and.it3.eq.0.and.it4.ne.0) then
        q3=q2+q4-q1
      else if(it1.ne.0.and.it2.ne.0.and.it3.ne.0.and.it4.eq.0) then
        q4=q1+q3-q2
      else if(it1.eq.0.and.it2.eq.0.and.it3.ne.0.and.it4.ne.0) then
        if(itype(m,n+2).ne.0) then
          q1=2.*q4-q(m,n+2)
        else
          q1=q4
        endif
        if(itype(m+1,n+2).ne.0) then  
          q2=2.*q3-q(m+1,n+2)
        else
          q2=q3
        endif
      else if(it1.ne.0.and.it2.eq.0.and.it3.eq.0.and.it4.ne.0) then
        if(itype(m-1,n).ne.0) then
          q2=2.*q1-q(m-1,n)
        else
          q2=q1
        endif
        if(itype(m-1,n+1).ne.0) then
          q3=2.*q4-q(m-1,n+1)
        else
          q3=q4
        endif
      else if(it1.ne.0.and.it2.ne.0.and.it3.eq.0.and.it4.eq.0) then
        if(itype(m+1,n-1).ne.0) then
          q3=2.*q2-q(m+1,n-1)
        else
          q3=q2
        endif
        if(itype(m,n-1).ne.0) then
          q4=2.*q1-q(m,n-1)
        else
          q4=q1
        endif
      else if(it1.eq.0.and.it2.ne.0.and.it3.ne.0.and.it4.eq.0) then
        if(itype(m+2,n+1).ne.0) then
          q4=2.*q3-q(m+2,n+1)
        else
          q4=q3
        endif
        if(itype(m+2,n).ne.0) then
          q1=2.*q2-q(m+2,n)
        else
          q1=q2
        endif
      else if(it1.ne.0.and.it2.eq.0.and.it3.ne.0.and.it4.eq.0) then
        if(itype(m-1,n).ne.0) then
          q2=2.*q1-q(m-1,n)
        else
          q2=(q1+q3)/2.d0
        endif
        if(itype(m+2,n+1).ne.0) then
          q4=2.*q3-q(m+2,n+1)
        else
          q4=(q1+q3)/2.d0
        endif
      else if(it1.eq.0.and.it2.ne.0.and.it3.eq.0.and.it4.ne.0) then
        if(itype(m+2,n).ne.0) then
          q1=2.*q2-q(m+2,n)
        else
          q1=(q2+q4)/2.d0
        endif
        if(itype(m-1,n+1).ne.0) then
          q3=2.*q4-q(m-1,n+1)
        else
          q3=(q2+q4)/2.d0
        endif
      else if(it1.ne.0.and.it2.eq.0.and.it3.eq.0.and.it4.eq.0) then
        if(itype(m-1,n).ne.0) then
          q2=2.*q1-q(m-1,n)
        else
          q2=q1
        endif
        if(itype(m,n-1).ne.0) then
          q4=2.*q1-q(m,n-1)
        else
          q4=q1
        endif
        q3=q2+q4-q1
      else if(it1.eq.0.and.it2.ne.0.and.it3.eq.0.and.it4.eq.0) then
        if(itype(m+2,n).ne.0) then
          q1=2.*q2-q(m+2,n)
        else
          q1=q2
        endif
        if(itype(m+1,n-1).ne.0) then
          q3=2.*q2-q(m+1,n-1)
        else
          q3=q2
        endif
        q4=q1+q3-q2
      else if(it1.eq.0.and.it2.eq.0.and.it3.ne.0.and.it4.eq.0) then
        if(itype(m+1,n+2).ne.0) then
          q2=2.*q3-q(m+1,n+2)
        else
          q2=q3
        endif
        if(itype(m+2,n+1).ne.0) then
          q4=2.*q3-q(m+2,n+1)
        else
          q4=q3
        endif
        q1=q2+q4-q3
      else if(it1.eq.0.and.it2.eq.0.and.it3.eq.0.and.it4.ne.0) then
        if(itype(m,n+2).ne.0) then
          q1=2.*q4-q(m,n+2)
        else
          q1=q4
        endif
        if(itype(m-1,n+1).ne.0) then
          q3=2.*q4-q(m-1,n+1)
        else
          q3=q4
        endif
        q2=q1+q3-q4
      endif
c     Bilinear interpolation 
      qint=(q1*(m+1-x)+q2*(x-m))*(n+1-y)+
     &         (q4*(m+1-x)+q3*(x-m))*(y-n)
      return
      end

c**********************************************************************
c  julday computes the julian day for a given date
c----------------------------------------------------------------------
      function julday(idd,imm,iyr)
      dimension js(12),je(12)
      ly=0
      if( ( ((iyr/4)*4.eq.iyr) .and. ((iyr/100)*100.ne.iyr) ) .or.
     &    ( (iyr/400)*400.eq.iyr) ) ly=1
      js(1)=1
      je(1)=31
      js(2)=32
      je(2)=59+ly
      k=1
      do i=3,12
        js(i)=je(i-1)+1
        je(i)=js(i)+29+k
        k=1-k
        if(i.eq.7) k=1
      enddo
      julday=js(imm)+idd-1
      return
      end     

c----------------------------------------------------------------------
c  jdiff gives the no of days from id1/im1/iy1 to id2/im2/iy2 
c      the day-difference may be negative and the years differ by 1
c----------------------------------------------------------------------
      function  jdiff(id1,im1,iy1,id2,im2,iy2)
      if(iy2.eq.iy1) jdiff = julday(id2,im2,iy2) 
     &      - julday(id1,im1,iy1)
      if(iy2.eq.iy1+1) jdiff = julday(id2,im2,iy2) 
     &      + julday(31,12,iy1) - julday(id1,im1,iy1)
      if(iy2.eq.iy1-1) jdiff = julday(id2,im2,iy2) 
     &      - julday(31,12,iy2) - julday(id1,im1,iy1)
      return
      end

c----------------------------------------------------------------------
c      rand computes random numbers between 0.0 and 1.0
c----------------------------------------------------------------------
      function randmedslik(ix)
      implicit real*8(a-h,o-z)
      integer a,p,ix,b15,b16,xhi,xalo,leftlo,fhi,k
      data a/16807/,b15/32768/,b16/65536/,p/2147483647/
      xhi=ix/b16
      xalo=(ix-xhi*b16)*a
      leftlo=xalo/b16
      fhi=xhi*a+leftlo
      k=fhi/b15
      ix=(((xalo-leftlo*b16)-p)+(fhi-k*b15)*b16)+k
      if(ix.lt.0) ix=ix+p
      randmedslik=dfloat(ix)*4.656612875d-10

      return
      end

c----------------------------------------------------------------------
c      calculates an approximately random first seed for rand
c      (modified by M. De Dominicis)
c----------------------------------------------------------------------
      subroutine seedmedslik(ix)
      real*8 hms
      character date*8, time*10
      call date_and_time(date,time)
      call date_and_time(DATE=date)
      call date_and_time(TIME=time)
      read(time,'(f10.3)') hms
      ix=int(hms*100)
      iz=ix-(ix/1000000)*1000000
      iy=1000000
      ix=0
      do k=1,6
        ix0=iz
        iz=iz/10
        ix1=ix0-iz*10
        iy=iy/10
        ix=ix+ix1*iy
      enddo
      write(90,*) 'Random seed = ',ix
      return
      end

c----------------------------------------------------------------------
c      reads satellite contour
c      (R. Lardner, M. De Dominicis, 2009)
c----------------------------------------------------------------------
      subroutine readsat(ix,ntot,px,py,experiment_folder)
      implicit real*8 (a-h,o-z)
      parameter(npc=100000)
      dimension segx(10000,2), segy(10000,2), px(npc), py(npc)
      character datestamp*10,aslik*2,empty*80,experiment_folder*400
      data datestamp /'0711020357'/, aslik /'01'/
      integer lines_counter, npcl_count
      open(38,file=
     &     TRIM(experiment_folder)//'/xp_files/contour_slick.csv')
      read(38,*) empty
      read(38,*) ndata
      read(38,*) empty
      nsegs = 1
      read(38,*) ystart, xstart
      segx(nsegs,1) = xstart
      segy(nsegs,1) = ystart
      ! Read points in the file
      lines_counter = 2
      do while (lines_counter.le.ndata)
        read(38,*) y, x
        segx(nsegs,2) = x
        segy(nsegs,2) = y
        if (x.eq.xstart.and.y.eq.ystart) then
          if (lines_counter.eq.ndata) exit
          read(38,*) y, x
          xstart = x
          ystart = y
          lines_counter = lines_counter+1
        endif
        nsegs = nsegs + 1
        segx(nsegs,1) = x
        segy(nsegs,1) = y
        lines_counter = lines_counter+1
      enddo
      ! Create box containing all segments,
      ! This is used for generating random numbers within a reasonable range.
      box_xmax = segx(1,1)
      box_ymax = segy(1,1)
      box_xmin = segx(1,1)
      box_ymin = segy(1,1)
      do i = 1, nsegs
          do j = 1, 2
              x_pnt = segx(i, j)
              y_pnt = segy(i, j)
              if (box_xmax.lt.x_pnt) box_xmax = x_pnt
              if (box_ymax.lt.y_pnt) box_ymax = y_pnt
              if (box_xmin.gt.x_pnt) box_xmin = x_pnt
              if (box_ymin.gt.y_pnt) box_ymin = y_pnt
          end do
      end do
      npcl_count = 1 ! Particles counter
      do while (npcl_count.le.ntot)
        randx = randmedslik(ix) * (box_xmax-box_xmin) + box_xmin
        randy = randmedslik(ix) * (box_ymax-box_ymin) + box_ymin
        ! Find intersections between random point and polygon
        intersections = 0
        do i = 1, nsegs
          if ((randx.ge.segx(i, 1).and.randx.le.segx(i, 2)).or.
     &        (randx.le.segx(i, 1).and.randx.ge.segx(i, 2))) then
            coeff = (segy(i, 2) - segy(i, 1)) /
     &              (segx(i, 2) - segx(i, 1))
            if (coeff.ne.0.0d0) then
              y_int = segy(i, 1) + coeff * (randx - segx(i, 1))
            else
              y_int = segy(i, 2)
            end if
            if (y_int.lt.randy) intersections = intersections + 1
          end if
        end do
        if (mod(intersections, 2) == 1) then
          px(npcl_count) = randx
          py(npcl_count) = randy
          npcl_count = npcl_count + 1
        end if
      end do
      close(38)
      return
      end

c----------------------------------------------------------------------
c       fetch calculation
c       (R. Lardner, M. De Dominicis, 2009)
c----------------------------------------------------------------------
      subroutine calcfetch(xavg,yavg,wdirstoke,fetch,preproc_folder)
      implicit real*8 (a-h,o-z)
      parameter(npts=400000, imx=700, jmx=300)
      dimension alon(npts), alat(npts), seg(npts,5)
      character dummy*80,a3*3,preproc_folder*400
      logical ex
      glon(x)=along1+(x-1.d0)*dlong
      glat(y)=alatg1+(y-1.d0)*dlatg
      pi = 4.d0 * datan(1.d0)
      degrad = pi / 180.d0  
      m1=int(xavg+0.5d0)           ! centre of slick
      n1=int(yavg+0.5d0)
      xavg_lon=glon(dfloat(m1))
      yavg_lat=glat(dfloat(n1))
c     read map points 
      open(1,file=TRIM(preproc_folder)//'/bnc_files/dtm.map')
      read(1,*) ncontours
      k = 1
      nseg = 0
      do ni=1,ncontours
        read(1,*) isle
        do i=1,isle
          read(1,*) alon(i), alat(i) 
        enddo 
      if(.not.(isle.lt.50)) then
        seg(k,1) = alon(1) 
        seg(k,2) = alat(1) 
        do i=2,isle
          seg(k,3) = alon(i) 
          seg(k,4) = alat(i)

          k = k + 1
          seg(k,1) = alon(i) 
          seg(k,2) = alat(i) 
        enddo 
      endif
      enddo
      nseg = k - 1
      close(1)
      cos_squared = dcos(yavg_lat * degrad) **2
      xi1  = xavg_lon
      eta1 = yavg_lat
      angledeg = wdirstoke
      angle = angledeg * degrad
      csangle = dcos(angle)
      snangle = dsin(angle)

      xi2 = xi1 + 50.d0 * snangle
      eta2 = eta1 + 50.d0 * csangle
      dmin = 100.d0
      nsmin = 0

      do 62 ns = 1,nseg   

        xx1 = seg(ns,1)
        yy1 = seg(ns,2)
        xx2 = seg(ns,3)
        yy2 = seg(ns,4)

        ddel  = (xx2-xx1)*(eta2-eta1)-(yy2-yy1)*(xi2-xi1)
        ddel1 = (eta2-eta1)*(xi1-xx1)-(xi2-xi1)*(eta1-yy1)
        ddel2 = (yy2-yy1)*(xi1-xx1)-(xx2-xx1)*(eta1-yy1)
        if(ddel.eq.0.d0) go to 62

        alam=ddel1/ddel
        alamp=ddel2/ddel
        if(alam.ge.0.d0.and.alam.le.1.d0.and.alamp.gt.0.d0) then
          xx=xx1+alam*(xx2-xx1)
          yy=yy1+alam*(yy2-yy1)
          dd1=dsqrt(cos_squared*(xx-xi1)*(xx-xi1) +
     &               (yy-eta1)*(yy-eta1))
          if(dd1.lt.dmin) then
            dmin=dd1
            nsmn=ns
          endif  
        endif

   62 enddo  !ns    

      fetch = dmin
      fetch = fetch*60*1852
      if(fetch.gt.20000) fetch=20000    


      return
      end

c-------------------------------------------------------
c ERROR HANDLING SUBROUTINE
c-------------------------------------------------------
      SUBROUTINE HANDLE_NF_ERR(errcode)
      include 'netcdf.inc'
      integer errcode
      if(errcode.ne.NF_NOERR) then
      print *, 'Error: ', nf_strerror(errcode)
      stop "HANDLE_NF_ERROR"
      endif
      end

c-------------------------------------------------------
c ERROR HANDLING SUBROUTINE
c-------------------------------------------------------
      subroutine handle_err(errcode)
      implicit none
      include 'netcdf.inc'
      integer errcode
      if(errcode.ne.0) then
        print *, 'Error: ', nf_strerror(errcode)
        stop 2
      end if
      end
