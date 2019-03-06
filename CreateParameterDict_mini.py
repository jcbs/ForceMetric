#!/usr/bin/python3

import os
import glob
from ForceMetric import ForceCurve, ParameterDict, IdentifyScanMode
from matplotlib import pyplot as plt
from pprint import pprint
import numpy as np
import seaborn as sb


media = '/media/jacob/MicroscopyData/Data/AFM/%s'
sb.set_style("whitegrid")
yes = ['y', 'Y', 'j', 'J', 'Yes', 'yes']
sample = 'Arabidopsis/131Y/0001/DEX/2019-01-15/Sample09_LG_7d24h/%s'
path = media % (sample % '*.ibw')

# The data of the images

data = np.sort(glob.glob(path))[:]
pprint(data)

# automatic recognition of Image and first next Force Curve

Modes = np.array([IdentifyScanMode(p) for p in data])
MaskScan = Modes == 'Imaging'
MaskForce = np.roll(MaskScan, 1)

img_files = data[MaskScan]
force_files = data[MaskForce]

sep = 'd'

img_no = np.array([
    os.path.basename(n).split('.')[0]
    for n in img_files
])


para_no = [
            os.path.basename(n)
            for n in force_files
]

path = media % (sample)


if os.path.exists(path % 'Parameters.npy'):
    check = input('This file already exists do you want to change it? [y,n] ')
    if check not in yes:
        print('Quit program')
        raise SystemExit

############################################################################
################### Calculation of contact parameters ######################
############################################################################

para_fc = [ForceCurve(path % (pn)) for pn in para_no]
img_files = [path % (n) for n in img_no]
pprint(img_files)

A1_free = []
A1_near = []
phi1_free = []
phi1_near = []
A2_free = []
A2_near = []
phi2_free = []
phi2_near = []
d0 = []
para = []

savepath = path % 'Parameters.npy'

print(os.path.isfile(savepath))
dictionary = ParameterDict(savepath)
print(len(para_fc))

i = -1
for fc in para_fc:
    i += 1
    for fn in img_files:
        print(fn, i)
        fc.trace = True
        aols = fc['AmpInvOLS']
        dols = fc['InvOLS']
        d0.append(fc.getData('Defl')[0]/dols)
        fc.surfaceidx = 0

        # this loop is if the force curves have a bad quality and therefore the
        # contactpoint can't be determined. In this case the user can enter a
        # number of pixels which should be skipped which usually is enough to
        # determine the contact point correctly. Just change the -1 in the next
        # line to a list of numbers where you want to aid the algorithm
        if i in [-1]:
            print('correction loop')
            print(fc.force.Trace().shape)
            if i == 7:
                offset = 5700
                fc.force.data[:offset] = fc.force.data[offset + 1]
            elif i in [1, 2, 3]:
                offset = 3000
                fc.force.data[:offset] = fc.force.data[offset + 1]
        else:
            fc.correct(method='fiv', stds=6)
        ind = fc.indentation.Trace()*1e6
        f = fc.force.Trace()
        p1 = fc.phase1.Trace()
        a1 = fc.amp1.Trace()/aols*1e3
        idx = fc.contactidx

        # The offsets are for the far (off0) and near correction (off1) in
        # pixels
        off0 = 30
        off1 = 10

        A1_free.append(a1[off0])
        A1_near.append(a1[idx-off1])
        phi1_free.append(p1[off0])
        phi1_near.append(p1[idx-off1])

        try:
            p2 = fc.phase2.Trace()
            a2 = fc.amp2.Trace()/aols*1e3
            A2_free.append(a2[off0])
            A2_near.append(a2[idx-off1])
            phi2_free.append(p2[off0])
            phi2_near.append(p2[idx-off1])
        except:
            A2_free.append(np.nan)
            A2_near.append(np.nan)
            phi2_free.append(np.nan)
            phi2_near.append(np.nan)

        E0 = fc.Young('sneddon', fmin=150e-9, fmax=300e-9, constant='force')
        para.append({'free A1': A1_free[-1], 'near A1': A1_near[-1],
                     'free A2': A2_free[-1], 'near A2': A2_near[-1],
                     'free Phi1': phi1_free[-1], 'near Phi1': phi1_near[-1],
                     'free Phi2': phi2_free[-1], 'near Phi2': phi2_near[-1],
                     'Defl offset': d0[-1], 'E0': E0})
        dictionary.AddFileInfo(os.path.basename(fn).split('.')[0], para[-1])
        key = list(dictionary.parameters.keys())[i]
        print("%s \n" % key)
        pprint(dictionary.parameters[key])

        print("Phase: %.2f \t %.2f" % (p1[off0], p1[idx-off1]))
        print("Amplitude: %.2f mV \t %.2f mV" % (a1[off0], a1[idx-off1]))
        try:
            print("Phase: %.2f \t %.2f" % (p1[off0], p2[idx-off1]))
            print("Amplitude: %.2f mV \t %.2f mV" % (a2[off0], a2[idx-off1]))
        except:
            print('No parameters for 2nd eigenmode available')
        print("d0: %.2f V" % d0[-1])
        print("E0: %.2e Pa" % E0)

    fig, ax = plt.subplots()
    ax.plot(ind, f*1e9, ind, p1, ind, a1)
    ax.set_title("Curve no %i" % i)
    try:
        plt.plot(ind, f*1e9, ind, p2, ind, a2)
    except:
        pass
    ax.set_ylim(-1.5, 320)
    fig.tight_layout()
    # plt.grid()
    plt.show()

############################################################################
#############End of Calculation of contact parameters ######################
############################################################################
dictionary.Write(savepath)

with open(path % 'filelist.txt', mode='w') as data:
    for f in img_files:
            data.write(f + '.ibw' + '\n')
# data.close()
