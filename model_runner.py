from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import copy
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pcse.util import Afgen
from pcse.base import ParameterProvider
from pcse.models import Wofost72_WLP_FD, Wofost72_PP
from pcse.fileinput import CABOFileReader
from pcse.util import WOFOST72SiteDataProvider
from pcse.fileinput import YAMLCropDataProvider
from pcse.fileinput import CABOWeatherDataProvider
from pcse.fileinput import YAMLAgroManagementReader
warnings.filterwarnings("ignore")

#模型运行模块
class ModelRunner(object):
    """for WOFOST runner
    """
    mydir = 'Share_Data_for_field/'
    
    def __init__(self, code, params, wdp, year,calendar_date,select_pars):
        self.code = code
        self.params = params
        self.params0=copy.deepcopy(self.params)
        self.wdp = wdp
        self.year = year
        self.calendar_date=calendar_date
        self.VB,self.VS=self.params['VERNBASE'],self.params['VERNSAT']
        agromanagement_file = self.mydir+'amgt/%d_%d.amgt'%(self.code,self.year)

        if os.path.exists(agromanagement_file):
            agro=YAMLAgroManagementReader(agromanagement_file) 
            self.agro=change_end(agro)
        else:
            agromanagement_file=CreatAgro(self.code,self.year,self.calendar_date,agromanagement_file)()
            agro=YAMLAgroManagementReader(agromanagement_file)
            self.agro=change_end(agro)
        #标定参数名和实际参数的对应关系字典,如果存在，一定要把'β_DVS', 'α_v'放最后，方便统一编码
        #下面之所以这么绕是为了可以选择任意组合参数进行标定
        par_dct=dict(zip(['β_IDEM','α_TSUM1','α_TSUM2','α_TDWI', 'α_SPAN', 'α_CVO', 'α_CVL', 'α_SLATB', 'α_AMAXTB', 'β_DVS', 'α_v'], \
                        ['IDEM','TSUM1','TSUM2',"TDWI", "SPAN", 'CVO', 'CVL', "SLATB", "AMAXTB", ['FLTB', 'FSTB', 'FOTB'],['FLTB', 'FSTB', 'FOTB']]))    
        parameter_names=[par_dct[a] for a in select_pars]
        my_par=[]
        for mp in parameter_names:
            if isinstance(mp,list):
                for mmp in mp:
                    if mmp in my_par:
                        continue
                    my_par.append(mmp)
            else:
                my_par.append(mp)
        self.parameter_names = my_par
        self.select_pars = select_pars
        self.par_dct = par_dct

   
    def __call__(self, x,df_flg=False,wofsim_flg=False):

        createVar = locals()
        createVar['β_DVS'],createVar['α_v']=0.0,1.0 #第一次要以动态变量形式给出，否则更新不进去
        
        for par,v in zip(self.select_pars,list(x)):
            if par in ['β_DVS', 'α_v']:
                createVar[par]=v if v!=createVar[par] else createVar[par]                
            elif par.endswith('TB'):
                createVar[self.par_dct[par]] = α_y(self.params0[self.par_dct[par]],v)
            elif par=='β_IDEM':
                agro_update = change_em(self.agro,v)   
                createVar[self.par_dct[par]] = None    
            else:  
                createVar[self.par_dct[par]] = self.params0[self.par_dct[par]]*v  

        β_DVS=createVar['β_DVS']
        α_v=createVar['α_v']
        createVar['FLTB'],createVar['FOTB'],createVar['FSTB']=change_FTB(α_v,β_DVS) #newly chnaged 20220729
    
        #对应生成模型参数list
        createVar = locals()
        new_pars=[createVar[p] for p in self.parameter_names ] #[TDWI,SPAN,CVO,SLATB,AMAXTB,FLTB,FSTB,FOTB]

        for parname, value in zip(self.parameter_names, new_pars):
            if parname=='IDEM':
                continue

        mydct=  dict(zip(self.parameter_names, new_pars))
        try:
            mydct.pop('IDEM')
        except:
            pass
        self.params.update(mydct)

        if 'β_IDEM' in self.select_pars:
            params1, wdp1, agro_update1=copy.deepcopy(self.params), copy.deepcopy(self.wdp), copy.deepcopy(agro_update)
            wofsim = Wofost72_PP(params1, wdp1, agro_update1)
        else:
            params1, wdp1, agro1=copy.deepcopy(self.params), copy.deepcopy(self.wdp), copy.deepcopy(self.agro)
            wofsim = Wofost72_PP(params1, wdp1, agro1)
        #wofsim = Wofost71_WLP_FD(self.params, self.wdp, self.agro1)
        if wofsim_flg:
            return wofsim

        wofsim.run_till_terminate()
        df=pd.DataFrame(wofsim.get_output()) 

        if df_flg:
            return df
        else:
            return list(df.LAI)+list(df.TWSO)

# y乘积型list参数
def α_y(lst,scale):
    X,Y=lst[::2],np.array(lst[1::2])*scale
    regular_list=[(x,y) for (x,y) in zip(X,Y)] # pair
    return [item for sublist in regular_list for item in sublist] # alone

# 分配系数
def change_FTB(α_v,β_DVS):
     #wheat
    if 0.950+β_DVS<0.65:
        β_DVS=-0.3
    if β_DVS>=1:
        β_DVS=0.99
    if 0.7*α_v>0.9999:
        α_v=1.42
    if α_v<0.5:
        α_v=0.5

    
    FLTB  =   [0.000, 0.650*α_v,
               0.100, 0.650*α_v,
               0.250, 0.700*α_v,
               0.500, 0.500*α_v,
               0.646, 0.700*α_v,
               0.950+β_DVS, 0.000,
               2.000, 0.000]
    FLTBs = Afgen(FLTB)
    FOTB  =   [0.000, 0.000,
               0.950+β_DVS, 0.000,
               1.000+β_DVS, 1.000,
               2.000, 1.000]          
    FOTBs = Afgen(FOTB)
    FSTB  =   [0.000, 1-0.650*α_v,
               0.100, 1-0.650*α_v,
               0.250, 1-0.700*α_v,
               0.500, 1-0.500*α_v,
               0.646, 1-0.700*α_v,
               0.950+β_DVS, 1.000,
               1.000+β_DVS, 0.000,
               2.000, 0.000] 
    return FLTB,FOTB,FSTB 

def get_dt(a):
    return dt.date(a.year,a.month,a.day)

def clean_nan(a):
    #实现将首尾为nan的模拟序列，前设为第一个实际值，后设为最后一个实际值
    n=len(a)/4
    b=[]
    for t in range(4):
        a1=a[int(t*n):int((t+1)*n)]
        head=True
        i0=np.nan
        for i in a1:
            if t==3: #DVS
                b.append(i)
                continue
            if i!=i and head:
                b.append(0)
                i0=i
            elif i0!=i0 and i==i:
                head=False
                b.append(i)
                i0=i
            elif i0==i0 and i!=i:
                tail=i0
                b.append(tail)
                i0=i
            elif i0!=i0 and i!=i:
                b.append(tail)
                i0=i                
            else:
                b.append(i)
                i0=i     
    return b

def change_em(agro,idem):
    agro1=copy.deepcopy(agro)
    idem=round(idem)
    date_s=list(agro1[0].keys())[0]
    date_em=list(agro1[0].values())[0]['CropCalendar']['crop_start_date']
    t_dict=agro1[0][date_s]['CropCalendar']
    t_dict.update({'crop_start_date':date_em+dt.timedelta(idem)})
    agro1[0][date_s].update({'CropCalendar':t_dict})
    return agro1

def change_end(agro,type='harvest'):
    agro1=copy.deepcopy(agro)
    date_s=list(agro1[0].keys())[0]
    t_dict=agro1[0][date_s]['CropCalendar']
    t_dict.update({'crop_end_type':'harvest'})
    agro1[0][date_s].update({'CropCalendar':t_dict})
    return agro1   

def GetPhenoData(file):
    df_station=pd.read_excel(file, parse_dates=True )
    df_station=df_station[['站名',	'年份',	'站号',	'经度',	'纬度',	'出苗',	'开花',	'成熟']]
    df_station['doy_em']=df_station['出苗'].map(lambda x:x.timetuple().tm_yday)
    df_station['doy_flw']=df_station['开花'].map(lambda x:x.timetuple().tm_yday)
    df_station['doy_mat']=df_station['成熟'].map(lambda x:x.timetuple().tm_yday)
    return df_station

def ModelData(mylon,mylat,TSUM=None):
    mydir='Share_Data_for_field/'
    src_file=mydir+'weather/Precipitation-Flux_C3S-glob-agric_AgERA5_20181225_final-v1.0.nc'
    ds_src = xr.open_mfdataset(src_file)
    pnt=ds_src.sel(lon=mylon, lat=mylat, method="nearest")
    w_lon,w_lat=round(pnt.lon.values.item()*10)/10,round(pnt.lat.values.item()*10)/10
    site="%.1f_%.1f"%(w_lon,w_lat)

    '''气象数据'''
    weatherfile =  mydir+'weather/%s'%site
    wdp = CABOWeatherDataProvider(weatherfile)

    '''作物参数,Winter_wheat_105是模型的小麦默认参数中的一个品种,临时先用这个默认参数'''
    cropfile = mydir+'cropdata' 
    cropd = YAMLCropDataProvider(cropfile,force_reload=True)
    cropd.set_active_crop('wheat', "Winter_wheat_105")

    '''土壤数据'''
    soilfile = mydir+'soildata/zhengzhou.soil'
    soild = CABOFileReader(soilfile)

    '''所在点的一些其他参数'''
    sited = WOFOST72SiteDataProvider(
            IFUNRN = 0,#Indicates whether non-infiltrating fraction of rain is a function of storm size (1)
            SSMAX  = 0.000000, #Maximum depth of water that can be stored on the soil surface [cm]
            WAV    = 20.000000, #Initial amount of water in total soil profile [cm]
            NOTINF = 0.000000,#Maximum fraction of rain not-infiltrating into the soil [0-1], default 0.
            SSI    = 0.000000, #Initial depth of water stored on the surface [cm]
            SMLIM  = 0.400000,# Initial maximum moisture content in initial rooting depth zone [0-1], default 0.4
            )# CO2    = 360. Atmospheric CO2 level (ppm), default 360.


    cropd1=dict(cropd)

    #春化参数修改，此处根据纬度近似一个结果
    VS=2.4665*mylat - 35.751
    VB=VS/5

    cropd1.update({'VERNBASE': VB,
            'VERNSAT': VS,
            'DLO': 15
            }) 
    if TSUM != None:
        cropd1.update({
        'TSUM1': TSUM[0],
        'TSUM2': TSUM[1]
        }) 

    params = ParameterProvider(cropdata=cropd1, sitedata=sited, soildata=soild)
    return wdp, params

class CreatAgro():    
    def __init__(self, code, year, calendar_date, outputfile, SM=0.15, irrigation=5.0):
        self.code = code
        self.year=year
        self.calendar_date=calendar_date
        self.outputfile = outputfile 
        self.SM=SM
        self.irrigation=irrigation
    
    def __call__(self):

        em,flw,mt=self.calendar_date
        file=self.outputfile

        f=open(file,'w+')
        f.write('Version: 1.0.0\n'
                # +'AgroManagement:\n- %s:\n'%em.strftime("%Y-%m-%d")
                +'AgroManagement:\n- %s:\n'%(em.strftime("%Y-")+'10-01') #相同的起始日期
                +'    CropCalendar:\n'
                +'        crop_name: wheat\n'
                +'        variety_name: Winter_wheat_105\n'
                +'        crop_start_date: %s\n'%em.strftime("%Y-%m-%d")
                +'        crop_start_type: emergence\n'
                # +'        crop_end_date: %s\n'%mt.strftime("%Y-%m-%d") 
                +'        crop_end_date: %s\n'%(mt.strftime("%Y-")+'07-01') #放宽统一结束时间，为了更好标定TSUM2                
                +'        crop_end_type: earliest\n'
                +'        max_duration: 330\n'
                +'    TimedEvents:\n'
                +'    StateEvents:\n'
                # # +'    -   event_signal: irrigate\n'
                # # +'        event_state: DVS\n'
                # # +'        zero_condition: rising\n'
                # # +'        name:  Soil moisture driven irrigation scheduling\n'
                # # +'        comment: All irrigation amounts in cm\n'
                # # +'        events_table:\n'
                # # +'        - 0.3: {amount:  %.1f, efficiency: 0.8}\n'%self.irrigation1                     
                # # +'        - 1.2: {amount:  %.1f, efficiency: 0.8}\n'%self.irrigation2
                # +'    -   event_signal: irrigate\n'
                # +'        event_state: SM\n'
                # +'        zero_condition: falling\n'
                # +'        name:  Soil moisture driven irrigation scheduling\n'
                # +'        comment: All irrigation amounts in cm\n'
                # +'        events_table:\n'
                # +'        - %.4f: {amount:  %.3f, efficiency: 0.7}\n'%(self.SM,self.irrigation)                    
                # +'- %s:'%mt.strftime("%Y-%m-%d")
                )
        f.close()

        return file

       
