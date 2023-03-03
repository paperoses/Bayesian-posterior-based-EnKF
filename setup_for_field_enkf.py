from __future__ import absolute_import   
from __future__ import division         
from __future__ import print_function     
from __future__ import unicode_literals  

import spotpy
import warnings
import rioxarray

import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
import scipy.stats as ss
from model_runner import ModelData,ModelRunner
from spotpy.parameter import Uniform,Normal

warnings.filterwarnings("ignore")

class spot_setup(object):
      
    def __init__(self, select_pars, area, i_th0, year,  cal_prior=True, obj_func=None):
        if cal_prior:
            createVar = locals() 
            createVar['β_IDEM']  = Normal("β_IDEM", mean=0 , stddev=1.5, step=1)       
            createVar['α_TSUM1']  = Normal("α_TSUM1", mean=1 , stddev=0.05, step=0.02) 
            createVar['α_TSUM2']  = Normal("α_TSUM2", mean=1 , stddev=0.05, step=0.02)           
            createVar['α_TDWI']  = Normal("α_TDWI", mean=1 , stddev=0.3, step=0.1)
            createVar['α_SPAN']  = Normal("α_SPAN", mean=1 , stddev=0.05, step=0.02)
            createVar['α_CVO']  = Normal("α_CVO", mean=1 , stddev=0.05)
            createVar['α_SLATB'] = Normal("α_SLATB", mean=1 , stddev=0.05, step=0.01)
            createVar['α_AMAXTB']    = Normal("α_AMAXTB", mean=1 , stddev=0.05, step=0.01)
            createVar['β_DVS']    = Normal('β_DVS', mean=0 , stddev=0.08, step=0.04) 
            createVar['α_v'] =  Normal('α_v', mean=1 , stddev=0.05, step=0.01) 
            self.pars=[]
            
            for p in select_pars:
                self.pars.append(createVar[p])   

            self.prior=[]
            for par in self.pars:  
                if type(par)==spotpy.parameter.Normal:
                    mu,sigma = par.rndargs
                    xmin,xmax = par.minbound, par.maxbound
                    truncnorm=ss.truncnorm((xmin - mu) / sigma, (xmax - mu) / sigma, loc=mu, scale=sigma)
                    self.prior.append(truncnorm)

                elif type(par)==spotpy.parameter.Uniform:
                    xmin,xmax = par.rndargs
                    uniform=ss.uniform(xmin, (xmax - xmin))
                    self.prior.append(uniform) 

        self.area_grids=gpd.read_file('Share_Data_for_field/%s_calibrated_DVS.shp'%area)

        self.ds_i_th=rioxarray.open_rasterio('Share_Data_for_field/Henan_calibrated_DVS_fid.tif')

        data = np.load('Share_Data_for_field/data.npy', allow_pickle=True)
        self.data=data

        self.mu_sigmas=[[i,i*0.025] for i in data[i_th0][7]] #LAI
        self.mu_sigmas.append([data[i_th0][5],data[i_th0][5]*0.1])
        self.lon,self.lat=data[i_th0][3]
        i_th=self.ds_i_th.sel(x=self.lon,y=self.lat,method='nearest').data[0]

        self.dates=[dt.date(year,1,1)+dt.timedelta(d-1) for d in range(49,146,8)]
        
        self.calendar_date = self.area_grids['Em_%d'%year][i_th], \
                             self.area_grids['Flw_%d'%year][i_th], \
                             self.area_grids['Mat_%d'%year][i_th]
        self.dates.append(dt.date(year,1,1)+dt.timedelta(int(self.calendar_date[2]-1))) 

        self.select_pars=select_pars
        self.area = area
        self.i_th=i_th
        self.year=year
        self.obj_func = obj_func

    def parameters(self):
        return spotpy.parameter.generate(self.pars) 

    def prior_dist(self,x):
        log_pri=0
        for i in range(len(x)):  
            log_pri+=self.prior[i].logpdf(x[i]) 
        return log_pri

    def simulation(self,x,return_df=False):    
        code=self.area_grids.Id[self.i_th]

        TSUM1,TSUM2=self.area_grids['TSUM1_%d'%self.year][self.i_th],self.area_grids['TSUM2_%d'%self.year][self.i_th]
        wdp, params=ModelData(self.lon,self.lat,TSUM=[TSUM1,TSUM2])
        model_runner = ModelRunner(code, params, wdp, self.year,self.calendar_date,self.select_pars)
        sim_prior=model_runner(x,df_flg=return_df)
        if not return_df:
            log_prior=self.prior_dist(x)
            sim_prior.append(log_prior)
        return sim_prior

    def evaluation(self,start=4,end=12,enkf_flg=False): 
        df_obs=pd.DataFrame(dict(zip(self.dates,list(self.mu_sigmas)))).T
        df_obs.columns=['mu','sigma']
        end=12
        arr=np.array(df_obs.mu)[:-1]
        start=list(arr).index(arr.max())-1 

        if enkf_flg:
            observations_for_DA=[]
            for i in range(start,end):
                d=self.dates[i]
                lai, errlai=self.mu_sigmas[i]
                observations_for_DA.append((d, {"LAI":(lai, errlai*2)}))
            return observations_for_DA
        
        index=list(range(start,end))+[-1]
        return df_obs.iloc[index]

    def objectivefunction(self,simulation,evaluation, params=None): 
        log_like = 0
        n=len(simulation[:-1])
        LAIs,TWSOs=simulation[:-1][:int(n/2)],simulation[:-1][int(n/2):]
        LAIs.append(np.nanmax(TWSOs))
        sims=LAIs
        sim_dates=[dt.date(self.year-1,10,1)+dt.timedelta(d) for d in range(int(n/2))]
        sim_dates.append(self.dates[-1])
        df_sim=pd.DataFrame([sim_dates,sims]).T
        df_sim.columns=['date','mu']
        df_sim=df_sim.set_index('date')
        df_sim=df_sim[~df_sim.index.duplicated(keep='last')] 

        df_obs=pd.DataFrame(evaluation['mu'])
        df_sigma=pd.DataFrame(evaluation['sigma'])
        diff=df_obs-df_sim
        diff=diff[diff.mu==diff.mu]
        d=len(diff)-1 
        index=list(range(d))+[-1]
        sigma=np.array(df_sigma['sigma'])[index]

        log_like=ss.multivariate_normal.logpdf(list(diff['mu']), mean=[0]*len(diff), cov=np.diag(sigma**2)) 
        log_pri=simulation[-1]

        return log_pri+log_like   
