import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set(color_codes=True)

## change experiment names here to plot

data_folder = 'logs'
if not os.path.isdir(data_folder):
    print('WARNING! LOG FOLDER NOT FOUND, MAKE SURE YOU HAVE CORRECT DATA FOLDER')
simple_colors = ['red','green','blue','purple']
colors = ['red','orange','yellow','green','blue','purple','black','grey','olive','cyan','pink','darkgreen','lightblue','plum']


# d1 = np.loadtxt(os.path.join(data_folder,'ip_bl_shallow_1000'))
# d2 = np.loadtxt(os.path.join(data_folder,'ip_bl_middle_1000'))
# d3 = np.loadtxt(os.path.join(data_folder,'ip_bl_deep_1000'))
#
# ax = sns.tsplot(data=d1,color='red',condition='ip_bl_shallow_1000')
# ax = sns.tsplot(data=d2,color='green',condition='ip_bl_middle_1000')
# ax = sns.tsplot(data=d3,color='blue',condition='ip_bl_deep_1000')

# d1 = np.loadtxt(os.path.join(data_folder,'ip_shallow_1000'))
# d2 = np.loadtxt(os.path.join(data_folder,'ip_middle_1000'))
# d3 = np.loadtxt(os.path.join(data_folder,'ip_deep_1000'))
#
# ax = sns.tsplot(data=d1,color='red',condition='ip_shallow_1000')
# ax = sns.tsplot(data=d2,color='green',condition='ip_middle_1000')
# ax = sns.tsplot(data=d3,color='blue',condition='ip_deep_1000')

# exp_names = ['ip1','ip04','ip01','ip004','ip_01_w','ip_03_w','ip_05_w']
# exp_names = ['ip_005_mw','ip_02_mw','ip_005_mw_wd','ip_02_mw_wd']
# exp_names = ['ip_005_mw','ip_02_mw']

exp_names = ['is_b1000_pop5_lr_5_sigma1','is_b1000_pop20_lr_5_sigma1',
             'is_b1000_pop50_lr_5_sigma1','is_b1000_pop100_lr_5_sigma1','is_b1000_pop500_lr_5_sigma1',
             'is_b3000_pop500_lr_5_sigma1','is_b1000_pop3000_lr_5_sigma1']


exp_names = ['es_pop5_lr_5_sigma1','es_pop20_lr_5_sigma1','es_pop50_lr_5_sigma1','es_pop100_lr_5_sigma1']

exp_names = ['is_b1000_pop50_lr_5_sigma_1','is_b1000_pop100_lr_5_sigma_1','is_b1000_pop500_lr_5_sigma_1','is_b1000_pop3000_lr_5_sigma_1']


exp_names = ['vanilla_pop10lr05sig05',
'vanilla_pop50lr05sig05',
'vanilla_pop100lr05sig05',
'full_pop10lr05sig05',
'full_pop50lr05sig05',
'full_pop100lr05sig05']


exp_names = ['v_swim_pop100_lr1_sig1','v_swim_pop100_lr1_sig05','v_swim_pop100_lr05_sig1','v_swim_pop100_lr05_sig05','v_swim_pop100_lr2_sig1','v_swim_pop100_lr1_sig2','v_swim_pop100_lr2_sig2']

exp_names = ['v64_hc_pop100_lr1_sig025','v64_hc_pop100_lr1_sig05','v64_hc_pop100_lr1_sig1','v64_hc_pop100_lr1_sig4']

exp_names = ['hc1','hc2','hc3','hc4','hc6','hc8','hc9']


exp_names = ['hc_es_vanilla_baseline','hc_es_vanilla_baseline_largepop', 'hc_fs', 'hc_ms_fs', 'hc_ms_fs_largepop']

exp_names = []
for i in range(14):
    new_name = "es"+str(i+1)
    exp_names.append(new_name)



# e  v_hc_pop100_lr05_sig005_fs
# e  v_hc_pop100_lr05_sig005_ms
# name  v_hc_pop100_lr05_sig005_ms_fs
# e  v_hc_pop100_lr05_sig025_fs
#   v_hc_pop100_lr05_sig025_ms
# ame  v_hc_pop100_lr05_sig025_ms_fs

exp_names = ['v_hc_pop100_lr05_sig005','v_hc_pop100_lr05_sig005_fs','v_hc_pop100_lr05_sig005_ms','v_hc_pop100_lr05_sig005_ms_fs']
exp_names = ['v_hc_pop100_lr05_sig025','v_hc_pop100_lr05_sig025_fs','v_hc_pop100_lr05_sig025_ms','v_hc_pop100_lr05_sig025_ms_fs']


"""
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32    --mode vanilla --exp_name  v_hc_pop100_lr05_sig005
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32  -fs    --mode vanilla --exp_name  v_hc_pop100_lr05_sig005_fs
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32  -ms    --mode vanilla --exp_name  v_hc_pop100_lr05_sig005_ms
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32  -ms  -fs  --mode vanilla --exp_name  v_hc_pop100_lr05_sig005_ms_fs
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32  -wd 0.99  --mode vanilla --exp_name  v_hc_pop100_lr05_sig005_wd99
python ES_IS.py --envname HalfCheetah-v2 -e 5 -n 150 -pop 100 -ep 100 -lr 0.5 -sigma 0.05 -hn  32  -ms  -fs -wd 0.99  --mode vanilla --exp_name  v_hc_pop100_lr05_sig005_ms_fs_wd99"""



exp_names = ['v_hc_pop100_lr05_sig005','v_hc_pop100_lr05_sig005_fs','v_hc_pop100_lr05_sig005_ms',
             'v_hc_pop100_lr05_sig005_ms_fs','v_hc_pop100_lr05_sig005_wd99','v_hc_pop100_lr05_sig005_ms_fs_wd99']

exp_names = ['center1','center2','center3','center4','center5']

exp_names = ['full_beta0_gamma09_pop50','full_beta01_gamma09_pop50',
             'full_beta02_gamma09_pop50','full_beta04_gamma09_pop50','full_beta075_gamma09_pop50','full_beta1_gamma09_pop50']

exp_names = ['v_hc_n100_pop100_lr05_sig0005','v_hc_n100_pop100_lr05_sig001','v_hc_n100_pop100_lr05_sig005',
             'v_hc_n100_pop100_lr05_sig02','v_hc_n100_pop100_lr05_sig05','v_hc_n100_pop100_lr05_sig1']


exp_names = ['vanilla',]
for i in range(10):
    new_name = "center"+str(i+40)
    exp_names.append(new_name)

exp_names = ['vanilla','full1_naive_best1','full2_naive_best05','full3_naive_best02']
exp_names = ['vanilla','full2_naive_best05','full4_best05_clip005','full5_best05_noclip']
exp_names = ['vanilla','full2_naive_best05','full6_best05_paramdistance']
exp_names = ['vanilla','full2_naive_best05','fullall_best05_pd_bl_beta03','fullall_best05_pd_bl_beta05','fullall_best05_pd_bl_beta08','fullall_best02_pd_bl_beta08']

exp_names = ['vanilla','full_nobl_best02_beta08','full_bl_best02_beta08_pd','full_bl_best02_beta08','full_bl_best02_beta09','full_bl_best01_beta09']

exp_names = ['full_hc_vanilla','full_hc_best02_beta06','full_hc_best02_beta09','full_hc_best05_beta06']

exp_names = []
for i in range(0,7+1):
    new_name = "pghc95nofix_single_epi60_"+str(i)
    exp_names.append(new_name)

title = 'pg hc lr comparison, n300, ep100, nep50, hd64'
exp_names = ['pghc_lr00001','pghc_lr00002','pghc_lr00005','pghc_lr0001','pghc_lr0002','pghc_lr0005','pghc_lr001','pghc_lr002','pghc_lr005']

title = 'pg hc hd comparison, n300, ep100, nep50, lr0.0005'
exp_names = ['pghc_lr00005_hn32','pghc_lr00005_hn128','pghc_lr00005_hn256',
             'pghc_lr0001_hn32','pghc_lr0001_hn128','pghc_lr0001_hn256']

# title = 'pg hc wd comparison, n300, ep100, nep50, hd64'
# exp_names = ['pghc_lr00005','pghc_lr00005_wd0001','pghc_lr00005_wd001','pghc_lr00005_wd002',
# 'pghc_lr00005_wd005','pghc_lr00005_wd01','pghc_lr00005_wd02',
# 'pghc_lr00005_wd05','pghc_lr00005_wd1']
#
title = 'pg hc lr comparison, n300, ep100, nep50, hd128'
exp_names = ['pghc_hd128_lr00001',
'pghc_hd128_lr00002','pghc_hd128_lr00005','pghc_hd128_lr0001',
'pghc_hd128_lr0002','pghc_hd128_lr0005','pghc_hd128_lr001','pghc_hd128_lr002','pghc_hd128_lr005', 'pghc_hd128_lr000001','pghc_hd128_lr000002','pghc_hd128_lr000005']

title = 'ppo hc lr comparison, n300, ep100, nep50, hd64'
exp_names = ['ppo_hd_64_lr00001','ppo_hd_64_lr00002','ppo_hd_64_lr00005',
             'ppo_hd_64_lr0001','ppo_hd_64_lr0002','ppo_hd_64_lr0005',
             'ppo_hd_64_lr001','ppo_hd_64_lr002','ppo_hd_64_lr005',]


title = 'pg hc lr comparison, share param, n300, ep100, nep50, hd64'
exp_names = ['pghc_hd64_pvs_lr00001','pghc_hd64_pvs_lr00002','pghc_hd64_pvs_lr00005',
'pghc_hd64_pvs_lr0001','pghc_hd64_pvs_lr0002','pghc_hd64_pvs_lr0005',
'pghc_hd64_pvs_lr001','pghc_hd64_pvs_lr002','pghc_hd64_pvs_lr005',]


title = 'pg hc gae comparison, n300, ep100, nep50, hd64'
exp_names = ['pghc_64_gaebt_1e-4','pghc_64_gaebt_2e-4','pghc_64_gaebt_5e-4','pghc_64_gaebt_1e-3',
             'pghc_64_nogae_1e-4','pghc_64_nogae_2e-4','pghc_64_nogae_5e-4','pghc_64_nogae_1e-3']


title = 'ppo scaled and unscaled compare'
exp_names = ['ppo_scale_baseline', 'ppo_noscale_baseline']



# exp_names = ['pghc_lr00005','pghc_lr00005_wd02']

label_names = exp_names ## use this if names are same

# exp_names = ['pghc1','pghc2','pghc3','pghc4']

#
# for i in range(1,11):
#     new_name = "full"+str(i)
#     exp_names.append(new_name)

## using data is 30k vs 60-70k for pop 50, 268k for pop 3000


"""
 ip_005_mw_wd
ip_02_mw_wd
"""

n = len(exp_names)
data_list = []
for i in range(n):
    data_list.append( np.loadtxt(os.path.join(data_folder,exp_names[i])))

for i in range(n):
    ax = sns.tsplot(data=data_list[i], color=colors[i], condition=exp_names[i])

plt.xlabel('Epoch')
plt.ylabel('Return')
plt.title(title)
plt.tight_layout()
plt.show()
