'''
MIT License

Copyright (c) 2021-2022 Lei Wu @ leiwugeoph@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import pandas as pd
import pygplates
import os
from past.utils import old_div
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
import cartopy.mpl.geoaxes
import warnings
warnings.filterwarnings('ignore')

def get_overlap(start1, end1, start2, end2):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

def select_Evans_poles(Evans_category='Apoles', age_interval=[500, 650]):
    df_Evans = pd.read_excel(os.getcwd()+'/PMbias_utils/Evans+_2021_PCpoles.xlsx', sheet_name=Evans_category, index_col=None)
    ind_Evans = []
    df_Evans['Category'] = np.nan # create an empty column with name "Name"
    craton_list = ['Baltica', 'Congo', 'Rio de la Plata', 'India', 'Laurentia', 'West Africa'] # , , 'Australia-N', 'Australia-S', 'South China', 'Tarim'
    for index, row in df_Evans.iterrows():
        if Evans_category=='Apoles': 
            if row['Q4']==0:
                df_Evans['Category'].iloc[index] = 'A'
            else:
                df_Evans['Category'].iloc[index] = 'Aq4'
        if Evans_category=='Bpoles': 
            if row['Q4']==0:
                df_Evans['Category'].iloc[index] = 'B'
            else:
                df_Evans['Category'].iloc[index] = 'Bq4'
        if get_overlap(row['Min'], row['Max'], age_interval[0], age_interval[1])>0:
            if row['Craton'] in craton_list: 
                ind_Evans.append(index)
    return df_Evans.loc[ind_Evans]

def create_E21poles_shp(df_Apoles, df_Bpoles, craton_name='E21_balt', model='R21', anchor_id=1, save_flag=0):
    from shapely.geometry import polygon
    df = pd.concat([df_Apoles, df_Bpoles])

    if craton_name == 'E21_Ama': 
        df = df[df['Craton']=='Amazonia']
    elif craton_name == 'E21_balt': 
        df = df[df['Craton']=='Baltica']
    elif craton_name == 'E21_lau': 
        df = df[df['Craton']=='Laurentia']
    elif craton_name == 'E21_SF_Con': 
        df = df[df['Craton']=='Congo']
    elif craton_name == 'E21_RDLPlata': 
        df = df[df['Craton']=='Rio de la Plata']
    elif craton_name == 'E21_waf': 
        df = df[df['Craton']=='West Africa']
    else: 
        df = pd.DataFrame(columns = df.columns)

    # create an empty column with name "Coordinates"
    df['Coordinates'], df['FROMAGE'], df['TOAGE'] = np.nan, np.nan, np.nan
    df['Rlon'], df['Rlat'], df['RecE'], df['RecEAge'] = np.nan, np.nan, np.nan, np.nan
    df['RlonMin'], df['RlatMin'], df['RecEMin'], df['RecEMinAge'] = np.nan, np.nan, np.nan, np.nan
    df['name'], df['plateid'], df['Plon'], df['Plat'], df['poleA95'] = np.nan, np.nan, np.nan, np.nan, np.nan

    for index,row in df.iterrows():
        df['name'], df['Plon'], df['Plat'], df['poleA95'] = df['Rockname_component'], df['PLONG'], df['PLAT'], df['A95']
        # call func create_ellipse to create circle boundaries
        X, Y = create_ellipse(float(row.PLONG), float(row.PLAT), float(row.A95), float(row.A95), .0, n=30)
        polygon = [(i,j) for i,j in zip(X,Y)]
        # print(polygon)
        df['Coordinates'][index] = Polygon(polygon) # call func Polygon to create polygon for shapefiles
        # df['Coordinates'][index] = polygon
        df.loc[index, 'name'] = row.name
        df['FROMAGE'][index] = float(row.Max)
        df['TOAGE'][index] = float(row.Min)

        # reconstruct paleopoles
        pole = pygplates.PointOnSphere(float(row.PLAT), float(row.PLONG))
        plateid_tmp = int(999)
        # print(model)
        # reconstruct sampling pole to formation age
        if model == 'R21':
            model_filename = os.getcwd()+'/PMbias_utils/Robert+_2021_ESR/ROTATIONS_IAPETUS_MODEL_Av.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 2201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 302
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 101
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 2203
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 7701 
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 7703
                # print('E21_waf')
        if model == 'S21':
            model_filename = os.getcwd()+'/PMbias_utils/Scotese_PaleoAtlas_v3/PALEOMAP_PlateModel.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 301
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 101
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 202
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 701 
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 714 
                # print('E21_waf')
        if model == 'M21': 
            model_filename = os.getcwd()+'/PMbias_utils/Merdith+_2021_ESR/1000_0_rotfile_Merdith_et_al.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 302
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 101
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 22041 
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 701 
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 714 
                # print('E21_waf')
        if model == 'TC16': 
            model_filename = os.getcwd()+'/PMbias_utils/CEED6/Torsvik_Cocks_HybridRotationFile.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 302
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 101
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 202
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 701 
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 714 
                # print('E21_waf')
        if model == 'M17': 
            model_filename = os.getcwd()+'/PMbias_utils/Merdith+_2017_GR/Neoproterozoic_rotations.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 2201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 3001
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 1001
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 2203
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 7701 
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 7703 
                # print('E21_waf')
        if model == 'L08': 
            model_filename = os.getcwd()+'/PMbias_utils/Li+_2008_PR/unified_ll.rot'
            if craton_name == 'E21_Ama': 
                plateid_tmp = 201
                # print('E21_Ama')
            elif craton_name == 'E21_balt':
                plateid_tmp = 302
                # print('E21_balt')
            elif craton_name == 'E21_lau':
                plateid_tmp = 199
                # print('E21_lau')
            elif craton_name == 'E21_RDLPlata': 
                plateid_tmp = 294
                # print('E21_RDLPlata')
            elif craton_name == 'E21_SF_Con': 
                plateid_tmp = 717
                # print('E21_SF_Con')
            elif craton_name == 'E21_waf': 
                plateid_tmp = 714
                # print('E21_waf')

        df['plateid'][index] = plateid_tmp

        # rotation_model = pygplates.RotationModel(model_filename)
        # point_rotation = rotation_model.get_rotation(float(row.Nominal_age), plateid_tmp, anchor_plate_id=anchor_id)
        # rec_pole = point_rotation * pole
        # df['Rlon'][index], df['Rlat'][index] = rec_pole.to_lat_lon()[0], rec_pole.to_lat_lon()[1] 
        # if haversine_distance(point1=rec_pole.to_lat_lon(), point2=[-90,0]) >= 90:
        #     df['RecE'][index] = haversine_distance(point1=rec_pole.to_lat_lon(), point2=[90,0])
        # else:
        #     df['RecE'][index] = haversine_distance(point1=rec_pole.to_lat_lon(), point2=[-90,0])
        df['Rlon'][index], df['Rlat'][index], df['RecE'][index] = cal_RecE(model_filename, pole, plateid_tmp, anchor_id, age=row.Nominal_age) 
        df['RecEAge'][index] = row.Nominal_age
        # create a list of ages at steps of 0.5 Myr
        t_step = .5
        ages = np.arange(float(row.Min), float(row.Max), t_step)
        ages = np.append(ages, ages[-1]+.5)
        res_all_tmp = []
        for age in list(ages):
            Rec_tmp = {}
            Rlon_tmp, Rlat_tmp, RecE_tmp = cal_RecE(model_filename, pole, plateid_tmp, anchor_id, age=age)
            # print(float(age), float(Rlon_tmp), float(Rlat_tmp), float(RecE_tmp)) 
            Rec_tmp['age'], Rec_tmp['Rlon_tmp'], Rec_tmp['Rlat_tmp'], Rec_tmp['RecE_tmp'] = float(age), float(Rlon_tmp), float(Rlat_tmp), float(RecE_tmp)
            res_all_tmp.append(Rec_tmp)

        df_Rec_tmp = pd.DataFrame.from_dict(res_all_tmp)
        df_Rec_tmp2 = df_Rec_tmp[df_Rec_tmp['RecE_tmp'] == df_Rec_tmp['RecE_tmp'].min()]
        if len(df_Rec_tmp2) == 1: 
            df_Rec_tmp1 = df_Rec_tmp2
        else:
            df_Rec_tmp1 = df_Rec_tmp2.iloc[0]
        # print(index, len(df_Rec_tmp1))
        # print(df_Rec_tmp1['Rlon_tmp'], df_Rec_tmp1['Rlat_tmp'], df_Rec_tmp1['RecE_tmp'], df_Rec_tmp1['age'])
        # df['RlonMin'][index], df['RlatMin'][index], df['RecEMin'][index], df['RecEMinAge'][index] = df_Rec_tmp1['Rlon_tmp'], df_Rec_tmp1['Rlat_tmp'], df_Rec_tmp1['RecE_tmp'], df_Rec_tmp1['age']
        df['RlonMin'][index], df['RlatMin'][index] = df_Rec_tmp1['Rlon_tmp'], df_Rec_tmp1['Rlat_tmp']
        df['RecEMin'][index], df['RecEMinAge'][index] = df_Rec_tmp1['RecE_tmp'], df_Rec_tmp1['age']  

    df = df.sort_values(by=['FROMAGE','TOAGE'], ascending=False)

    # df['Coordinates'] = Polygon(df['Coordinates'])
    # print(df.info())
    gdf_tmp = gpd.GeoDataFrame(df, geometry='Coordinates') 
    # gdf_tmp = df
    # create a smaller subset of the geodataframe (columns can be rearranged based on your preference; can add more colums if you want)
    gdf = gdf_tmp[['name', 'Category', 'Plon', 'Plat', 'poleA95', 'FROMAGE','TOAGE', 'plateid', 'Rlon', 'Rlat', 'RecE', 'RecEAge', 'RlonMin', 'RlatMin', 'RecEMin', 'RecEMinAge', 'Coordinates','GPMDB_result#']]

    if save_flag == 1 & len(gdf) > 0:
        # print(len(gdf))
        gdf.to_file(f'./Robert_2021_ESR/{model}_poles_{craton_name}.shp') # save to a shapefile
    return gdf

def cal_RecE(model_filename, pole, plateid_tmp, anchor_id, age):
    rotation_model = pygplates.RotationModel(model_filename) 
    point_rotation = rotation_model.get_rotation(float(age), plateid_tmp, anchor_plate_id=anchor_id)
    rec_pole = point_rotation * pole
    Rlon_tmp, Rlat_tmp = rec_pole.to_lat_lon()[0], rec_pole.to_lat_lon()[1] 
    if haversine_distance(point1=rec_pole.to_lat_lon(), point2=[-90,0]) >= 90:
        RecE_tmp = haversine_distance(point1=rec_pole.to_lat_lon(), point2=[90,0])
    else:
        RecE_tmp = haversine_distance(point1=rec_pole.to_lat_lon(), point2=[-90,0])
    return Rlon_tmp, Rlat_tmp, RecE_tmp

def create_gdf(df_Apoles, df_Bpoles, mname='R21', anchor_id=1, save_flagn=0):
    gdf_E21_Ama = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_Ama', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    gdf_E21_balt = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_balt', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    gdf_E21_lau = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_lau', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    gdf_E21_RDLPlata = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_RDLPlata', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    gdf_E21_SF_Con = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_SF_Con', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    gdf_E21_waf = create_E21poles_shp(df_Apoles, df_Bpoles, 'E21_waf', model=mname, anchor_id=anchor_id, save_flag=save_flagn)
    return gdf_E21_Ama, gdf_E21_balt, gdf_E21_lau, gdf_E21_RDLPlata, gdf_E21_SF_Con, gdf_E21_waf

def plot_pole_rec_err_E21(ax, gdf, title='Baltica', xylim=[500, 700, 0, 90], MinRec='N'):
    gdf_A = gdf[gdf['Category']=='A'].sort_values(by=['RecEMinAge'], ascending=False)
    gdf_B = gdf[gdf['Category']=='B'].sort_values(by=['RecEMinAge'], ascending=False)
    gdf_Aq4 = gdf[gdf['Category']=='Aq4'].sort_values(by=['RecEMinAge'], ascending=False)
    gdf_Bq4 = gdf[gdf['Category']=='Bq4'].sort_values(by=['RecEMinAge'], ascending=False)
    ssize = 150 
    if MinRec == 'N':
        ages = [(a+b)/2 for a, b in zip(gdf.FROMAGE.tolist(), gdf.TOAGE.tolist())]
        ages_B = [(a+b)/2 for a, b in zip(gdf_B.FROMAGE.tolist(), gdf_B.TOAGE.tolist())]
        ages_Bq4 = [(a+b)/2 for a, b in zip(gdf_Bq4.FROMAGE.tolist(), gdf_Bq4.TOAGE.tolist())]
        ages_A = [(a+b)/2 for a, b in zip(gdf_A.FROMAGE.tolist(), gdf_A.TOAGE.tolist())]
        ages_Aq4 = [(a+b)/2 for a, b in zip(gdf_Aq4.FROMAGE.tolist(), gdf_Aq4.TOAGE.tolist())]

        h2 = plt.bar(ages, gdf.poleA95, width=3., color='skyblue', label='Pole error')
        h1, = plt.plot(ages, gdf.RecE, 'k--', label='Rec error')
        # plt.scatter(ages, gdf.RecE, s=ssize, marker='o',color='None', edgecolors='m')
        h3 = plt.scatter(ages_B, gdf_B.RecE, s=ssize, marker='s',color='None', edgecolors='k', label='B')
        h4 = plt.scatter(ages_Bq4, gdf_Bq4.RecE, s=ssize, marker='s',color='g', edgecolors='k', label='Bq4')
        h5 = plt.scatter(ages_A, gdf_A.RecE, s=ssize, marker='o',color='None', edgecolors='m', label='A')
        h6 = plt.scatter(ages_Aq4, gdf_Aq4.RecE, s=ssize, marker='o',color='m', edgecolors='k', label='Aq4')        
    else:
        ages = gdf.RecEMinAge.tolist()
        ages_B = gdf_B.RecEMinAge.tolist()
        ages_Bq4 = gdf_Bq4.RecEMinAge.tolist()
        ages_A = gdf_A.RecEMinAge.tolist()
        ages_Aq4 = gdf_Aq4.RecEMinAge.tolist()

        h2 = plt.bar(ages, gdf.poleA95, width=3., color='skyblue', label='Pole error')
        h1, = plt.plot(ages, gdf.RecEMin, 'k--', label='Rec error')
        # plt.scatter(ages, gdf.RecEMin, s=ssize, marker='o',color='None', edgecolors='m')
        h3 = plt.scatter(ages_B, gdf_B.RecEMin, s=ssize, marker='s',color='None', edgecolors='k', label='B')
        h4 = plt.scatter(ages_Bq4, gdf_Bq4.RecEMin, s=ssize, marker='s',color='g', edgecolors='k', label='Bq4')
        h5 = plt.scatter(ages_A, gdf_A.RecEMin, s=ssize, marker='o',color='None', edgecolors='m', label='A')
        h6 = plt.scatter(ages_Aq4, gdf_Aq4.RecEMin, s=ssize, marker='o',color='m', edgecolors='k', label='Aq4')

    # plt.text(505, 80, int(gdf[gdf['poleA95'] < gdf['RecEMin']].RecEMin.sum()))
    plt.legend(handles=[h1, h2, h3, h4, h5, h6])
    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[2], xylim[3])
    plt.xlabel('Age (Ma)')
    plt.ylabel('Distance from the South Pole (deg)')
    plt.grid(color='lightgrey', linestyle='--', linewidth=1) # plt.grid(True)
    # plt.title(f'Reconstruction quality for {continent_name}');
    err = gdf[gdf['poleA95'] < gdf['RecEMin']]
    errA = err[(err['Category']=='A') | (err['Category']=='Aq4')]
    errQ4 = err[(err['Category']=='Bq4') | (err['Category']=='Aq4')]
    tmp, tmpA, tmpQ4 = int(err.RecEMin.sum()), int(errA.RecEMin.sum()), int(errQ4.RecEMin.sum())
    plt.title(f'{title} (Err_A/Err_q4/Err: {tmpA}/{tmpQ4}/{tmp})');

def plot_pole_rec_err_E21_all(gdf_E21_Ama, gdf_E21_balt, gdf_E21_lau, gdf_E21_RDLPlata, gdf_E21_SF_Con, gdf_E21_waf, MinRec0='N'):
    fig = plt.figure(figsize=(18, 10))
    plot_pole_rec_err_E21(plt.subplot(2,3,1), gdf=gdf_E21_Ama, title='Amazonia', MinRec=MinRec0)
    plot_pole_rec_err_E21(plt.subplot(2,3,2),gdf=gdf_E21_balt, title='Baltica', MinRec=MinRec0)
    plot_pole_rec_err_E21(plt.subplot(2,3,3),gdf=gdf_E21_lau, title='Laurentia', MinRec=MinRec0)
    plot_pole_rec_err_E21(plt.subplot(2,3,4),gdf=gdf_E21_RDLPlata, title='Rio De La Plata', MinRec=MinRec0)
    plot_pole_rec_err_E21(plt.subplot(2,3,5),gdf=gdf_E21_SF_Con, title='Sao Fransisco-Congo', MinRec=MinRec0)
    plot_pole_rec_err_E21(plt.subplot(2,3,6),gdf=gdf_E21_waf, title='West Africa', MinRec=MinRec0);

def rec_gdf_row(row, rotation_model, recon_age=0, anchor=1, antipole='N'):
    import pygplates
    plateid, Oldage, Youngage = row.plateid, row.FROMAGE, row.TOAGE
    # print(row.Coordinates.xy.dtype)
    try:
        lat, long = row.Coordinates.exterior.coords.xy[1], row.Coordinates.exterior.coords.xy[0]
    except:
        lat, long = row.Coordinates.xy[1], row.Coordinates.xy[0]
    ori_shp = pygplates.Feature()
    # if hasattr(row.Coordinates, 'geom_type'):
    if row.Coordinates.geom_type == 'Polygon':
        polygon = pygplates.PolygonOnSphere(zip(lat,long))
        ori_shp.set_geometry(polygon)
    ori_shp.set_valid_time(Oldage, Youngage)
    ori_shp.set_reconstruction_plate_id(int(plateid))
    rec_shp = []
    if recon_age <= Oldage and recon_age >= Youngage:
        pygplates.reconstruct(ori_shp, rotation_model, rec_shp, recon_age, anchor)
        rec_ls = rec_shp[0].get_reconstructed_geometry().to_lat_lon_list()
        if antipole == 'Y': rec_ls = [(-lat, lon-180) for lat, lon in rec_ls]
        df_rec_shp = pd.DataFrame(rec_ls, columns=['Plat','Plong'])
    else:
        df_rec_shp = pd.DataFrame([np.nan, np.nan], columns=['Plat','Plong'])
    return ori_shp, rec_shp, df_rec_shp

def plot_R21_poles_rec(gdf, rotation_model, fcolor='m', recon_age=0, anchor=1):
    for i, row in gdf.iterrows():
        if recon_age <= row.FROMAGE and recon_age >= row.TOAGE:
            df_rec_shp = []
            ori_shp, rec_shp, df_rec_shp = rec_gdf_row(row, rotation_model, recon_age=recon_age, anchor=anchor)
            plt.plot(df_rec_shp.Plong, df_rec_shp.Plat, color=fcolor, linestyle='-', linewidth=1, alpha=.8, transform=ccrs.Geodetic()) # Geodetic, PlateCarree

def antipode(PlatR, PlonR):
    return -PlatR, 180-PlonR

def plot_E21_poles_rec(gdf, rotation_model, fcolor='m', recon_age=0, anchor=1):
    # gdf_A = gdf[gdf['Category']=='A']
    # gdf_B = gdf[gdf['Category']=='B']
    # gdf_Aq4 = gdf[gdf['Category']=='Aq4']
    # gdf_Bq4 = gdf[gdf['Category']=='Bq4']
    ssize = 80
    alphaV = 1.
    # NA_centroid = [47, -90]
    for i, row in gdf.iterrows():
        if recon_age <= row.FROMAGE and recon_age >= row.TOAGE:
            PlatR, PlonR, rec_pole = [], [], []
            df_rec_shp = []
            point_rotation = rotation_model.get_rotation(float(recon_age), int(row.plateid), anchor_plate_id=anchor)
            rec_pole = point_rotation * pygplates.PointOnSphere(float(row.Plat), float(row.Plon)) 
            PlatR, PlonR = rec_pole.to_lat_lon()[0], rec_pole.to_lat_lon()[1] 

            # tmp_NA_centroid = pygplates.PointOnSphere(float(NA_centroid[0]), float(NA_centroid[1]))
            # PlatRa, PlonRa = antipode(PlatR, PlonR)
            # rec_poleA = pygplates.PointOnSphere(float(PlatRa), float(PlonRa)) 
            # gcd1 = pygplates.GreatCircleArc(rec_pole, tmp_NA_centroid).get_arc_length() * 180 / np.pi
            # gcd2 = pygplates.GreatCircleArc(rec_poleA, tmp_NA_centroid).get_arc_length() * 180 / np.pi
            # if gcd1 > gcd2:
            #     PlatR, PlonR = antipode(PlatR, PlonR)
            #     ori_shp, rec_shp, df_rec_shp = rec_gdf_row(row, rotation_model, recon_age=recon_age, anchor=anchor, antipole='Y') # recon A95
            # else:
            ori_shp, rec_shp, df_rec_shp = rec_gdf_row(row, rotation_model, recon_age=recon_age, anchor=anchor, antipole='N') # recon A95

            if row['Category']=='Aq4':
                plt.scatter(PlonR, PlatR, color=fcolor, s=ssize, marker='o', edgecolors='k', 
                                alpha=alphaV, transform=ccrs.PlateCarree()) # Geodetic, PlateCarree               
                plt.plot(df_rec_shp.Plong, df_rec_shp.Plat, color=fcolor, linestyle='-', linewidth=1, alpha=.8, transform=ccrs.Geodetic()) # Geodetic, PlateCarree
            elif row['Category']=='A':
                plt.scatter(PlonR, PlatR, color='none', s=ssize, marker='o', edgecolors=fcolor, 
                                alpha=alphaV, transform=ccrs.PlateCarree()) # Geodetic, PlateCarree  
                plt.plot(df_rec_shp.Plong, df_rec_shp.Plat, color=fcolor, linestyle='--', linewidth=1, alpha=.8, transform=ccrs.Geodetic()) # Geodetic, PlateCarree
            elif row['Category']=='Bq4':
                plt.scatter(PlonR, PlatR, color=fcolor, s=ssize, marker='s', edgecolors='k', 
                                alpha=alphaV, transform=ccrs.PlateCarree()) # Geodetic, PlateCarree, RotatedPole 
                plt.plot(df_rec_shp.Plong, df_rec_shp.Plat, color=fcolor, linestyle='-', linewidth=1, alpha=.8, transform=ccrs.Geodetic()) # Geodetic, PlateCarree
            elif row['Category']=='B':
                plt.scatter(PlonR, PlatR, color='none', s=ssize, marker='s', edgecolors=fcolor, 
                                alpha=alphaV, transform=ccrs.PlateCarree()) # Geodetic, PlateCarree 
                plt.plot(df_rec_shp.Plong, df_rec_shp.Plat, color=fcolor, linestyle='--', linewidth=1, alpha=.8, transform=ccrs.Geodetic()) # Geodetic, PlateCarree

def load_R21_model_E21(df_Apoles, df_Bpoles, map_axis, model='R21', recon_age=0, anchor_plate_id=1, graticule=0, plotpoles=0):
    import pygplates
    if model == 'R21':
        folder_rec = os.getcwd()+'/PMbias_utils/Robert+_2021_ESR' 
        input_coastlines_filename = folder_rec + '/CONTOURS_IAPETUS_MODEL.gpml'
        # rotation_filename = folder_rec + '/ROTATIONS_IAPETUS_MODEL.rot' 
        rotation_filename = folder_rec + '/ROTATIONS_IAPETUS_MODEL_Av.rot'
    elif model == 'M21': 
        folder_rec = os.getcwd()+'/PMbias_utils/Merdith+_2021_ESR' 
        input_coastlines_filename = folder_rec + '/shapes_continents_Merdith_et_al.gpml'
        rotation_filename = folder_rec + '/1000_0_rotfile_Merdith_et_al.rot' 
    elif model == 'S21': 
        folder_rec = os.getcwd()+'/PMbias_utils/Scotese_PaleoAtlas_v3' 
        input_coastlines_filename = folder_rec + '/PALEOMAP_PlatePolygons.gpml'
        rotation_filename = folder_rec + '/PALEOMAP_PlateModel.rot' 
    elif model == 'M17': 
        folder_rec = os.getcwd()+'/PMbias_utils/Merdith+_2017_GR' 
        input_coastlines_filename = folder_rec + '/Neoproterozoic_shapes.gpml'
        rotation_filename = folder_rec + '/Neoproterozoic_rotations.rot' 
        # anchor_plate_id=0
    elif model == 'TC16': 
        folder_rec = os.getcwd()+'/PMbias_utils/CEED6' 
        input_coastlines_filename = folder_rec + '/CEED6_LAND.gpml'
        rotation_filename = folder_rec + '/Torsvik_Cocks_HybridRotationFile.rot'
    elif model == 'L08': 
        folder_rec = os.getcwd()+'/PMbias_utils/Li+_2008_PR' 
        input_coastlines_filename = folder_rec + '/unified.gpml'
        rotation_filename = folder_rec + '/unified_ll.rot'
        # anchor_plate_id=0 

    rotation_model = pygplates.RotationModel(rotation_filename)
    output_coastlines_filename = folder_rec + f'/tmp{int(recon_age)}.shp'
    # output_coastlines_filename = folder_rec + f'/recon/tmp{int(recon_age)}.shp'

    # print(output_coastlines_filename[:])
    pygplates.reconstruct(input_coastlines_filename, rotation_model, output_coastlines_filename, recon_age, anchor_plate_id)
    output_coastlines_filename_OLD = output_coastlines_filename
    output_coastlines_filename_NEW = output_coastlines_filename[:-4] + f'/tmp{int(recon_age)}_polygon.shp'
    output_coastlines_filename_NEW1 = output_coastlines_filename[:-4] + f'/tmp{int(recon_age)}_polyline.shp'

    if plotpoles == 1:
        # print(model, anchor_plate_id)
        gdf_E21_Ama, gdf_E21_balt, gdf_E21_lau, gdf_E21_RDLPlata, gdf_E21_SF_Con, gdf_E21_waf = create_gdf(df_Apoles, df_Bpoles, mname=model, anchor_id=anchor_plate_id, save_flagn=0)
        # plot poles 
        # List of colors: https://matplotlib.org/stable/gallery/color/named_colors.html
        plot_E21_poles_rec(gdf_E21_Ama, rotation_model, fcolor='c', recon_age=recon_age, anchor=anchor_plate_id)
        plot_E21_poles_rec(gdf_E21_balt, rotation_model, fcolor='b', recon_age=recon_age, anchor=anchor_plate_id)
        plot_E21_poles_rec(gdf_E21_lau, rotation_model, fcolor='forestgreen', recon_age=recon_age, anchor=anchor_plate_id)
        plot_E21_poles_rec(gdf_E21_RDLPlata, rotation_model, fcolor='salmon', recon_age=recon_age, anchor=anchor_plate_id)
        plot_E21_poles_rec(gdf_E21_SF_Con, rotation_model, fcolor='orange', recon_age=recon_age, anchor=anchor_plate_id)
        plot_E21_poles_rec(gdf_E21_waf, rotation_model, fcolor='mediumpurple', recon_age=recon_age, anchor=anchor_plate_id)

    if graticule == 1:
        # print(anchor_plate_id)
        input_graticule = os.getcwd()+'/PMbias_utils/ne_10m_graticules_30/ne_10m_graticules_30.shp'
        output_graticule = os.getcwd()+'/PMbias_utils/ne_10m_graticules_30/ne_10m_graticules_30tmp.shp'
        pygplates.reconstruct(input_graticule, rotation_model, output_graticule, recon_age, anchor_plate_id)
        map_axis.add_geometries(gpd.read_file(output_graticule).geometry, ccrs.PlateCarree(), 
                                edgecolor='lightgrey', facecolor='none', alpha =1., linestyle='--', hatch='') # lightcoral, mistyrose, lightgrey
        # plot South Pole in the recon map
        pole = pygplates.PointOnSphere(-90, .0)
        point_rotation = rotation_model.get_rotation(float(recon_age), moving_plate_id=int(gpd.read_file(output_graticule).PLATEID.unique()[0]), anchor_plate_id=anchor_plate_id)
        # point_rotation = rotation_model.get_rotation(float(recon_age), moving_plate_id=0, anchor_plate_id=anchor_plate_id)
        rec_pole = point_rotation * pole
        map_axis.scatter(rec_pole.to_lat_lon()[1], rec_pole.to_lat_lon()[0] , color='darkgrey', s=50, marker='*',edgecolors='k', transform=ccrs.PlateCarree())
        # print('Y')

    return rotation_model, output_coastlines_filename_NEW, output_coastlines_filename_OLD, output_coastlines_filename_NEW1

def plot_global_recon(df_Apoles, df_Bpoles, mname='R21', projtype='orth', savefigflag=0):
    # [Amazonia, Baltica, Laurentia, RDLPlata, Sao Fransisco-Congo, West Africa]
    if mname == 'R21': plateids = [2201, 302, 101, 2203, 7701, 7703]
    elif mname == 'S21': plateids = [201, 301, 101, 202, 701, 714] 
    elif mname == 'M21': plateids = [201, 302, 101, 22041, 701, 714]
    elif mname == 'M17': plateids = [2201, 3001, 1001, 2203, 7701, 7703]
    elif mname == 'L08': plateids = [201, 302, 199, 294, 717, 714]
    elif mname == 'TC16': plateids = [201, 302, 101, 202, 701, 714] 

    cmap_plateids = pd.DataFrame({'plateids': plateids, 'cmapind': [ 'c', 'b', 'forestgreen', 'salmon', 'orange', 'mediumpurple']})

    # recon_ages = np.arange(650, 520, -40.)
    recon_ages = [650, 610, 590, 570, 560, 530]

    fig = plt.figure(figsize=(15, 10))
    # equdist, ortho, mollw, robin
    # proj, subplots_grid, camera_Angle, anchor_id = 'ortho', [2, 3, 0], [-160, -45], 1
    if projtype == 'orth': proj, subplots_grid, camera_Angle, anchor_id = 'ortho', [2, 3, 0], [-30, 45], 101
    # proj, subplots_grid, camera_Angle, anchor_id = 'mollw', [3, 2, 0], [-30, -45], 1
    if projtype == 'moll': proj, subplots_grid, camera_Angle, anchor_id = 'mollw', [3, 2, 0], [0, 0], 101

    if mname == 'M17': anchor_id = 1001
    elif mname == 'L08': anchor_id = 199
    # print(anchor_id)
    j = 0
    for k, recon_age in enumerate(recon_ages):

        subplots_grid[2] = k + 1
        
        # initiate map projection and plot recon
        if proj != 'ortho': #590: 
            map_axis = ini_proj_type_cart(proj_type=proj, view=[camera_Angle[0],0], subplots_grid=subplots_grid)
        else:
            if recon_age == 590: 
                camera_Angle_tmp = [camera_Angle[0],camera_Angle[1]] 
                # camera_Angle_tmp[0] = camera_Angle_tmp[0] + 180
                map_axis = ini_proj_type_cart(proj_type=proj, view=camera_Angle_tmp, subplots_grid=subplots_grid)
            else:
                map_axis = ini_proj_type_cart(proj_type=proj, view=camera_Angle, subplots_grid=subplots_grid)

        rotation_model, output_coastlines_filename_NEW, output_coastlines_filename_OLD, output_coastlines_filename_NEW1 = load_R21_model_E21(df_Apoles, df_Bpoles, map_axis, model=mname, recon_age=recon_age, anchor_plate_id=anchor_id, graticule=1, plotpoles=1)
        if mname == 'L08':
            # map_axis.add_geometries(Reader(output_coastlines_filename_NEW).geometries(), ccrs.PlateCarree(), 
            #                         edgecolor='grey', facecolor='lightgrey', alpha =.5, hatch='')
            plate_select1, plate_select_not1 = rec_plate_select(plateids, output_coastlines_filename_NEW1)
            for i in range(len(plate_select1)):
                j = cmap_plateids['cmapind'].where(cmap_plateids['plateids'] == plate_select1.iloc[i].PLATEID1).dropna().unique()
                # print(j)
                map_axis.add_geometries(ShapelyFeature([plate_select1.iloc[i].geometry], ccrs.PlateCarree()).geometries(), ccrs.PlateCarree(), 
                        edgecolor=j[0], facecolor='none', alpha =1., hatch='')
            map_axis.add_geometries(plate_select_not1.geometry, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha =1., hatch='')
        else:
            try:
                # map_axis.add_geometries(Reader(output_coastlines_filename_OLD).geometries(), ccrs.PlateCarree(), 
                #                         edgecolor='grey', facecolor='lightgrey', alpha =.5, hatch='')
                plate_select_OLD, plate_select_not_OLD = rec_plate_select(plateids, output_coastlines_filename_OLD)
                for i in range(len(plate_select_OLD)):
                    j = cmap_plateids['cmapind'].where(cmap_plateids['plateids'] == plate_select_OLD.iloc[i].PLATEID1).dropna().unique()
                    # print(j)
                    map_axis.add_geometries(ShapelyFeature([plate_select_OLD.iloc[i].geometry], ccrs.PlateCarree()).geometries(), ccrs.PlateCarree(), 
                            edgecolor=j[0], facecolor='none', alpha =1., hatch='')
                map_axis.add_geometries(plate_select_not_OLD.geometry, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha =1., hatch='')
            except:
                # map_axis.add_geometries(Reader(output_coastlines_filename_NEW).geometries(), ccrs.PlateCarree(), 
                #                         edgecolor='grey', facecolor='lightgrey', alpha =.5, hatch='')
                plate_select, plate_select_not = rec_plate_select(plateids, output_coastlines_filename_NEW)
                for i in range(len(plate_select)):
                    j = cmap_plateids['cmapind'].where(cmap_plateids['plateids'] == plate_select.iloc[i].PLATEID1).dropna().unique()
                    # print(j)
                    map_axis.add_geometries(ShapelyFeature([plate_select.iloc[i].geometry], ccrs.PlateCarree()).geometries(), ccrs.PlateCarree(), 
                            edgecolor=j[0], facecolor='none', alpha =1., hatch='')
                map_axis.add_geometries(plate_select_not.geometry, ccrs.PlateCarree(), edgecolor='grey', facecolor='none', alpha =1., hatch='')

        plt.title('%s Ma' % recon_age)
        
    fig.show()
    if savefigflag == 1: plt.savefig(f'./Global_Rec_{mname}.pdf')

def gmodel_err(gdf):
    err = gdf[gdf['poleA95'] < gdf['RecEMin']]
    errA = err[(err['Category']=='A') | (err['Category']=='Aq4')]
    errQ4 = err[(err['Category']=='Bq4') | (err['Category']=='Aq4')] 
    if len(errA.RecEMin) > 0: tmpA, tmpAn = (errA.RecEMin.sum()), (errA.RecEMin.sum()/len(errA.RecEMin))
    else: tmpA, tmpAn = 0, 0 
    if len(errQ4.RecEMin) > 0: tmpQ4, tmpQ4n = (errQ4.RecEMin.sum()), (errQ4.RecEMin.sum()/len(errQ4.RecEMin))
    else: tmpQ4, tmpQ4n = 0, 0  
    if len(err.RecEMin) > 0: tmp, tmpn = (err.RecEMin.sum()), (err.RecEMin.sum()/len(err.RecEMin))
    else: tmp, tmpn = 0, 0  
    return tmpA, tmpQ4, tmp, tmpAn, tmpQ4n, tmpn, len(errA.RecEMin), len(errQ4.RecEMin), len(err.RecEMin) 

def gmodel_err_all(gdf_E21_Ama_R21, gdf_E21_balt_R21, gdf_E21_lau_R21, gdf_E21_RDLPlata_R21, gdf_E21_SF_Con_R21, gdf_E21_waf_R21):
    err_all = []
    err_all.append(gmodel_err(gdf_E21_Ama_R21))
    err_all.append(gmodel_err(gdf_E21_balt_R21))
    err_all.append(gmodel_err(gdf_E21_lau_R21))
    err_all.append(gmodel_err(gdf_E21_RDLPlata_R21))
    err_all.append(gmodel_err(gdf_E21_SF_Con_R21))
    err_all.append(gmodel_err(gdf_E21_waf_R21))
    err_all = np.asarray(err_all)
    return gmodel_err(gdf_E21_waf_R21)[2], sum(err_all[:,0]), sum(err_all[:,1]), sum(err_all[:,2]), gmodel_err(gdf_E21_waf_R21)[5], sum(err_all[:,0])/sum(err_all[:,6]), sum(err_all[:,1])/sum(err_all[:,7]), sum(err_all[:,2])/sum(err_all[:,8]), err_all

def gmodel_err_all_breakdown(gdf_E21_Ama_R21, gdf_E21_balt_R21, gdf_E21_lau_R21, gdf_E21_RDLPlata_R21, gdf_E21_SF_Con_R21, gdf_E21_waf_R21):
    result =[gmodel_err(gdf_E21_balt_R21)[2], gmodel_err(gdf_E21_lau_R21)[2], gmodel_err(gdf_E21_RDLPlata_R21)[2], gmodel_err(gdf_E21_SF_Con_R21)[2], gmodel_err(gdf_E21_waf_R21)[2]]
    return result 

def create_ellipse(centerlon, centerlat, major_axis, minor_axis, angle, n=100):
    """
    This function enables general error ellipses

    Parameters
    -----------
    centerlon : longitude of the center of the ellipse
    centerlat : latitude of the center of the ellipse
    major_axis : Major axis of ellipse
    minor_axis : Minor axis of ellipse
    angle : angle of major axis in degrees east of north
    n : number of points with which to apporximate the ellipse

    Returns
    ---------

    """
    angle = angle * (np.pi/180)
    glon1 = centerlon
    glat1 = centerlat
    # major_axis = major_axis * (np.pi/180)
    # minor_axis = minor_axis * (np.pi/180)
    major_axis = major_axis
    minor_axis = minor_axis
    X = []
    Y = []
    for azimuth in np.linspace(-180, 180, n):
        az_rad = azimuth*(np.pi/180)
        radius = ((major_axis*minor_axis)/(((minor_axis*np.cos(az_rad-angle))
                                            ** 2 + (major_axis*np.sin(az_rad-angle))**2)**.5))

        # glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius* (180/np.pi)) # LW
        # X.append((360+glon2) % 360)
        X.append(glon2) # LW
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])
    # xy=pd.DataFrame({'X': X, 'Y': Y}) # LW for debug
    # print(xy) # LW for debug
    return X, Y

def shoot(lon, lat, azimuth, maxdist=None):
    """
    This function enables A95 error ellipses to be drawn around
    paleomagnetic poles in conjunction with equi
    (from: http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/)
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    scaleLW1 = .96
    # s = old_div(maxdist, 1.852)
    s = maxdist / scaleLW1 # LW
    faz = azimuth * np.pi / 180.

    EPS = 0.00000000005

    a = old_div(6378.13, 1.852)
    # a = old_div(6378.13, scaleLW1) # LW
    f = old_div(1, 298.257223563)
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf == 0):
        b = 0.
    else:
        b = 2. * np.arctan2(tu, cf)

    cu = old_div(1., np.sqrt(1 + tu * tu))
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (old_div(1., (r * r)) - 1.))
    x = old_div((x - 2.), x)
    c = 1. - x
    c = old_div((x * x / 4. + 1.), c)
    d = (0.375 * x * x - 1.) * x
    tu = old_div(s, (r * a * c))
    y = tu
    c = y + 1

    sy = np.sin(y)
    cy = np.cos(y)
    cz = np.cos(b + y)
    e = 2. * cz * cz - 1.
    c = y
    x = e * cy
    y = e + e - 1.
    y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
         d / 4. - cz) * sy * d + tu

    while (np.abs(y - c) > EPS):
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
             d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2 * np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2 * np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 = glon2 * 180/np.pi # LW
    glat2 = glat2 * 180/np.pi # LW
    baz = baz * 180/np.pi # LW

    return (glon2, glat2, baz)

def haversine_distance(point1=[0,0], point2=[90,0]):
    # calculate haversine / great-circle distance between points in [lat, long]
    pt1, pt2 = pygplates.PointOnSphere(point1), pygplates.PointOnSphere(point2)
    return np.degrees(pygplates.GreatCircleArc(pt1, pt2).get_arc_length())


def ini_proj_type_cart(proj_type='robin', view=[0,0], subplots_grid=[],
                       add_land=False, land_color='lightgrey', add_ocean=False, ocean_color='lightblue', 
                       grid_lines=True, lat_grid=np.arange(-90,90,30), lon_grid=np.arange(-180,180,30)):
    import cartopy
    
    # lat_gridm, lon_gridm = np.arange(-90,90,30), np.arange(-180,180,30)
    if proj_type == 'mollw':
        map_projection = ccrs.Mollweide(central_longitude=view[0])
    elif proj_type == 'ortho':
        map_projection = ccrs.Orthographic(central_longitude=view[0], central_latitude=view[1])
    elif proj_type == 'equdist':
        map_projection = ccrs.PlateCarree(central_longitude=view[0])
    elif proj_type == 'robin':
        map_projection = ccrs.Robinson(central_longitude=view[0])
        
    if subplots_grid:
        map_axis = plt.subplot(subplots_grid[0], subplots_grid[1], subplots_grid[2], projection=map_projection)
    else:
        map_axis = plt.axes(projection=map_projection)

    if add_ocean == True:
        map_axis.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor=ocean_color)
    if add_land == True:
        map_axis.add_feature(cartopy.feature.LAND, zorder=0,
                       facecolor=land_color, edgecolor='black')
    map_axis.set_global()
    if grid_lines == True:
        map_axis.gridlines(xlocs=lon_grid, ylocs=lat_grid)
    return map_axis

def rec_plate_select(plateid_q, output_coastlines_filename_old):
    # rotation_model, output_coastlines_filename, output_coastlines_filename_old = load_Wu_model(recon_age=age, anchor_plate_id=4)
    df = gpd.read_file(output_coastlines_filename_old)
    # df['PLATEID1'].loc[df.PLATEID1.isin(plateid_q)]
    # https://www.geeksforgeeks.org/selecting-rows-in-pandas-dataframe-based-on-conditions/
    plate_select = df.loc[df.PLATEID1.isin(plateid_q)]
    plate_select_not = df.loc[~df.PLATEID1.isin(plateid_q)]
    return plate_select, plate_select_not