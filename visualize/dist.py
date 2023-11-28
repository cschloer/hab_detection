import numpy as np

train_dist = [
    1.00000000e00,
    2.14136554e02,
    2.74848189e02,
    3.17455282e02,
    3.51229975e02,
    3.80320280e02,
    4.04333480e02,
    4.27179750e02,
    4.46079971e02,
    4.62208441e02,
    4.78737494e02,
    4.92992999e02,
    5.05092857e02,
    5.16277743e02,
    5.25563461e02,
    5.37672975e02,
    5.46104325e02,
    5.53554416e02,
    5.59509919e02,
    5.66738524e02,
    5.69482150e02,
    5.76403389e02,
    5.80302216e02,
    5.83583140e02,
    5.88942085e02,
    5.90772940e02,
    5.93926098e02,
    5.97525663e02,
    5.98049444e02,
    5.99569140e02,
    6.00489081e02,
    6.00943485e02,
    6.01417315e02,
    6.04076969e02,
    6.00604824e02,
    5.99459226e02,
    5.95482544e02,
    5.96610674e02,
    5.93200040e02,
    5.94012410e02,
    5.91274882e02,
    5.87658055e02,
    5.83077287e02,
    5.81105863e02,
    5.78556061e02,
    5.79363328e02,
    5.74068839e02,
    5.72048953e02,
    5.67976075e02,
    5.63955519e02,
    5.61098885e02,
    5.58452531e02,
    5.56506541e02,
    5.52260535e02,
    5.47551936e02,
    5.41973766e02,
    5.42847665e02,
    5.39006969e02,
    5.32446031e02,
    5.30555627e02,
    5.26180321e02,
    5.21268599e02,
    5.18669494e02,
    5.14562696e02,
    5.09945700e02,
    5.05277231e02,
    5.03666134e02,
    4.97533109e02,
    4.92121180e02,
    4.89567754e02,
    4.85034731e02,
    4.83903187e02,
    4.78605104e02,
    4.75537253e02,
    4.72135583e02,
    4.70204305e02,
    4.67187584e02,
    4.64860393e02,
    4.63212898e02,
    4.58681015e02,
    4.57302860e02,
    4.54279267e02,
    4.54495281e02,
    4.49056132e02,
    4.46897896e02,
    4.45665128e02,
    4.44282568e02,
    4.41842596e02,
    4.42445495e02,
    4.41913547e02,
    4.40301217e02,
    4.39005349e02,
    4.35428162e02,
    4.35744354e02,
    4.36638343e02,
    4.34932544e02,
    4.35209307e02,
    4.31709024e02,
    4.33591999e02,
    4.34128770e02,
    4.35183823e02,
    4.34726698e02,
    4.36117522e02,
    4.37111460e02,
    4.34438053e02,
    4.38054562e02,
    4.34728874e02,
    4.34962764e02,
    4.38088394e02,
    4.36737688e02,
    4.40335536e02,
    4.38899148e02,
    4.41301166e02,
    4.42611348e02,
    4.44593408e02,
    4.47323102e02,
    4.49106923e02,
    4.51668232e02,
    4.57012772e02,
    4.59669271e02,
    4.61907800e02,
    4.67997024e02,
    4.73643471e02,
    4.76727738e02,
    4.78102097e02,
    4.82559657e02,
    4.87629068e02,
    4.93833692e02,
    5.00156303e02,
    5.05227451e02,
    5.08545803e02,
    5.17207114e02,
    5.22743451e02,
    5.29579123e02,
    5.35161499e02,
    5.47510303e02,
    5.52219938e02,
    5.60669242e02,
    5.67048158e02,
    5.72363711e02,
    5.83299500e02,
    5.90759128e02,
    6.03200462e02,
    6.14049432e02,
    6.19224582e02,
    6.30952272e02,
    6.41571959e02,
    6.51271176e02,
    6.69328567e02,
    6.85860631e02,
    6.99016271e02,
    7.08740432e02,
    7.29950663e02,
    7.44601081e02,
    7.59082568e02,
    7.81914483e02,
    8.04809539e02,
    8.31207666e02,
    8.42537357e02,
    8.78392503e02,
    9.01671973e02,
    9.29546356e02,
    9.47871335e02,
    9.84223540e02,
    1.01981020e03,
    1.05659551e03,
    1.09454627e03,
    1.11465342e03,
    1.16296887e03,
    1.19497323e03,
    1.24995328e03,
    1.29738963e03,
    1.34926239e03,
    1.42747387e03,
    1.47928073e03,
    1.57273697e03,
    1.63882623e03,
    1.71359700e03,
    1.84375397e03,
    1.98201356e03,
    2.16493669e03,
    2.27935298e03,
    2.47071804e03,
    2.72445463e03,
    2.95642472e03,
    3.29539605e03,
    3.55923974e03,
    3.95322477e03,
    4.39140740e03,
    4.75223741e03,
    5.39086839e03,
    6.17472034e03,
    7.12707663e03,
    7.95216554e03,
    9.45725966e03,
    1.05830450e04,
    1.18302289e04,
    1.32122555e04,
    1.50488290e04,
    1.60114325e04,
    1.76401547e04,
    2.00902683e04,
    2.20115670e04,
    2.50003473e04,
    2.64541879e04,
    2.73093263e04,
    2.96960257e04,
    2.91491559e04,
    3.14137502e04,
    3.49603619e04,
    3.96627845e04,
    3.66416449e04,
    4.66545020e04,
    4.76510992e04,
    4.70334475e04,
    4.81313399e04,
    5.50786425e04,
    5.75916583e04,
    6.09002764e04,
    5.61036820e04,
    6.23511287e04,
    6.75551384e04,
    8.11197938e04,
    8.62980817e04,
    9.07499221e04,
    9.77729216e04,
    9.60099694e04,
    9.17565236e04,
    1.08201830e05,
    1.15860301e05,
    1.20376293e05,
    1.33770749e05,
    1.48603968e05,
    1.68331433e05,
    2.12369240e05,
    2.50675380e05,
    2.84725324e05,
    3.55979587e05,
    4.69984548e05,
    7.66121448e05,
    1.03789717e06,
    1.27033300e06,
    1.53055540e06,
    2.21649810e06,
    3.24706614e06,
    4.41188669e06,
    6.23203725e06,
    7.85166275e06,
    1.63499330e07,
    3.08832068e07,
    2.77948861e08,
    1.38974431e09,
    1.38974431e09,
    1.38974431e09,
]

test_dist = [
    1.00000000e00,
    3.00496318e02,
    3.86323755e02,
    4.41682167e02,
    4.88746301e02,
    5.23447895e02,
    5.55619059e02,
    5.89039171e02,
    6.11594157e02,
    6.39206758e02,
    6.57615843e02,
    6.76426923e02,
    6.83626414e02,
    7.10762112e02,
    7.32678694e02,
    7.45521717e02,
    7.47381080e02,
    7.57438892e02,
    7.79474495e02,
    7.83900580e02,
    7.89288213e02,
    8.03800068e02,
    8.16183314e02,
    8.25017074e02,
    8.20841614e02,
    8.37316928e02,
    8.39762810e02,
    8.30778381e02,
    8.40834343e02,
    8.51996030e02,
    8.61717412e02,
    8.62894124e02,
    8.55570963e02,
    8.61567001e02,
    8.56823535e02,
    8.56910538e02,
    8.64911078e02,
    8.55761275e02,
    8.59026499e02,
    8.45565902e02,
    8.60156273e02,
    8.51086885e02,
    8.52940489e02,
    8.40180842e02,
    8.45943227e02,
    8.45710777e02,
    8.49434343e02,
    8.49329549e02,
    8.38554213e02,
    8.28688869e02,
    8.36529762e02,
    8.30833788e02,
    8.36741134e02,
    8.22790755e02,
    8.17517247e02,
    8.03291632e02,
    8.04846109e02,
    7.99047007e02,
    7.91561766e02,
    7.92484941e02,
    7.76321762e02,
    7.77891465e02,
    7.67937877e02,
    7.55762755e02,
    7.55957121e02,
    7.45753365e02,
    7.51121301e02,
    7.52004384e02,
    7.39691153e02,
    7.42481180e02,
    7.24537594e02,
    7.23635730e02,
    7.20975348e02,
    7.20556344e02,
    7.08298919e02,
    7.13036433e02,
    7.04061375e02,
    7.05675667e02,
    7.01371567e02,
    6.86547143e02,
    6.87850458e02,
    6.93122760e02,
    6.76538878e02,
    6.62302408e02,
    6.51476329e02,
    6.62101263e02,
    6.41822531e02,
    6.42435646e02,
    6.33427656e02,
    6.34137021e02,
    6.26180425e02,
    6.15440037e02,
    6.17301776e02,
    6.07540563e02,
    6.03341303e02,
    6.03035335e02,
    5.86949756e02,
    6.00156432e02,
    5.81403253e02,
    5.92045201e02,
    5.86669393e02,
    5.76063323e02,
    5.61146599e02,
    5.73561352e02,
    5.62171570e02,
    5.56313799e02,
    5.56657082e02,
    5.46225697e02,
    5.47290707e02,
    5.46852545e02,
    5.37896781e02,
    5.29127937e02,
    5.37819374e02,
    5.28352104e02,
    5.28928953e02,
    5.19144371e02,
    5.13881381e02,
    5.17681492e02,
    5.13378172e02,
    5.11038508e02,
    5.03805935e02,
    5.07770193e02,
    4.99862025e02,
    4.92005719e02,
    4.89666602e02,
    4.85430683e02,
    4.91545345e02,
    4.80279619e02,
    4.79932469e02,
    4.72812887e02,
    4.67566838e02,
    4.75039111e02,
    4.69500184e02,
    4.67538427e02,
    4.60030543e02,
    4.63086697e02,
    4.65116193e02,
    4.68962382e02,
    4.66806780e02,
    4.64253650e02,
    4.66810944e02,
    4.63215430e02,
    4.65253505e02,
    4.70212413e02,
    4.79885809e02,
    4.78837045e02,
    4.84522632e02,
    4.81814311e02,
    4.89052379e02,
    4.93737499e02,
    5.00845798e02,
    5.00225226e02,
    5.05746074e02,
    5.16051684e02,
    5.22101325e02,
    5.33522022e02,
    5.36550867e02,
    5.36303380e02,
    5.53930827e02,
    5.63019692e02,
    5.81303777e02,
    5.94268295e02,
    6.04081079e02,
    6.30030414e02,
    6.42681851e02,
    6.71947862e02,
    6.83001736e02,
    6.94421690e02,
    7.19446652e02,
    7.34920145e02,
    7.70641175e02,
    7.96053911e02,
    8.22790755e02,
    8.64213922e02,
    8.92341974e02,
    9.20707080e02,
    9.71169518e02,
    1.03670259e03,
    1.07939113e03,
    1.12282305e03,
    1.20251706e03,
    1.24121094e03,
    1.30263151e03,
    1.39252125e03,
    1.43185658e03,
    1.46760982e03,
    1.57144103e03,
    1.58837652e03,
    1.73101826e03,
    1.76563465e03,
    1.87180290e03,
    1.95042675e03,
    1.96744195e03,
    2.15128192e03,
    2.17863897e03,
    2.19846708e03,
    2.25293698e03,
    2.32973960e03,
    2.48665737e03,
    2.61005865e03,
    2.61624486e03,
    2.73454853e03,
    2.74427288e03,
    2.81178836e03,
    2.81082161e03,
    2.83390156e03,
    2.84196746e03,
    2.82795854e03,
    2.93751026e03,
    2.98027784e03,
    3.03493995e03,
    3.21047528e03,
    3.29806093e03,
    3.34550928e03,
    3.36855591e03,
    3.36387796e03,
    3.53965456e03,
    3.84317535e03,
    3.96382959e03,
    4.16660756e03,
    4.26052129e03,
    4.23747168e03,
    4.41809719e03,
    4.68698561e03,
    5.00581292e03,
    5.54714253e03,
    6.36177202e03,
    7.10642406e03,
    8.48387563e03,
    9.42904267e03,
    1.02147042e04,
    1.30050104e04,
    1.47718119e04,
    1.62288331e04,
    2.13662846e04,
    2.73477722e04,
    3.30147386e04,
    6.28414098e04,
    8.23445984e04,
    1.31726480e05,
    2.21327233e05,
    2.72509155e05,
    1.22821028e06,
    1.28239602e06,
    2.75377673e06,
    7.47453683e06,
    3.27010986e07,
    2.61608789e08,
    2.61608789e08,
    7.47453683e06,
    1.06779098e06,
    2.61608789e08,
    2.61608789e08,
    2.61608789e08,
]

all_dist = [
    1.00000000e00,
    2.24350947e02,
    2.88014211e02,
    3.32259859e02,
    3.67616156e02,
    3.97540709e02,
    4.22560763e02,
    4.46621989e02,
    4.66061442e02,
    4.83414494e02,
    5.00296325e02,
    5.15123036e02,
    5.26891769e02,
    5.39671631e02,
    5.50203074e02,
    5.62517789e02,
    5.70441784e02,
    5.78211142e02,
    5.85693789e02,
    5.92752626e02,
    5.95766201e02,
    6.03448494e02,
    6.08145827e02,
    6.11953592e02,
    6.16535868e02,
    6.19678638e02,
    6.22810154e02,
    6.25340094e02,
    6.26717305e02,
    6.29096720e02,
    6.30782437e02,
    6.31304275e02,
    6.31117771e02,
    6.34099054e02,
    6.30472251e02,
    6.29417068e02,
    6.26394947e02,
    6.26675209e02,
    6.23779893e02,
    6.23392765e02,
    6.22081427e02,
    6.17959387e02,
    6.13845062e02,
    6.10950847e02,
    6.09053781e02,
    6.09787412e02,
    6.05146824e02,
    6.03248772e02,
    5.98573986e02,
    5.94018315e02,
    5.91976825e02,
    5.89045699e02,
    5.87687500e02,
    5.82607469e02,
    5.77778208e02,
    5.71422425e02,
    5.72364582e02,
    5.68306659e02,
    5.61568211e02,
    5.59870859e02,
    5.54484165e02,
    5.50013588e02,
    5.46786650e02,
    5.41964233e02,
    5.37665000e02,
    5.32478636e02,
    5.31400672e02,
    5.25715755e02,
    5.19675666e02,
    5.17493428e02,
    5.11838451e02,
    5.10706649e02,
    5.05527654e02,
    5.02612844e02,
    4.98465109e02,
    4.97019453e02,
    4.93490102e02,
    4.91427918e02,
    4.89547417e02,
    4.84137011e02,
    4.82946395e02,
    4.80510488e02,
    4.79422649e02,
    4.73192702e02,
    4.70293971e02,
    4.70005113e02,
    4.67055607e02,
    4.64835728e02,
    4.64638889e02,
    4.64205385e02,
    4.62028871e02,
    4.59891882e02,
    4.56746831e02,
    4.56179929e02,
    4.56625585e02,
    4.55027264e02,
    4.53794723e02,
    4.51797940e02,
    4.51788051e02,
    4.53282520e02,
    4.53744847e02,
    4.52307162e02,
    4.52074745e02,
    4.54230606e02,
    4.50659784e02,
    4.53320845e02,
    4.50356212e02,
    4.49466783e02,
    4.52388447e02,
    4.51128604e02,
    4.53362289e02,
    4.51084980e02,
    4.54214738e02,
    4.54290461e02,
    4.56114661e02,
    4.57346681e02,
    4.58257809e02,
    4.60980676e02,
    4.65102545e02,
    4.67107659e02,
    4.68074586e02,
    4.73877349e02,
    4.77612154e02,
    4.79084525e02,
    4.79897606e02,
    4.83012222e02,
    4.88245323e02,
    4.91635673e02,
    4.96839559e02,
    4.99799214e02,
    5.01581602e02,
    5.10034693e02,
    5.13517804e02,
    5.18675583e02,
    5.21664539e02,
    5.32141463e02,
    5.36308749e02,
    5.43821844e02,
    5.48392367e02,
    5.51999759e02,
    5.61117120e02,
    5.66067126e02,
    5.76138352e02,
    5.85667616e02,
    5.91993590e02,
    6.00720090e02,
    6.10236739e02,
    6.16899083e02,
    6.32398009e02,
    6.46035923e02,
    6.57784595e02,
    6.64836882e02,
    6.82050097e02,
    6.95783801e02,
    7.08160803e02,
    7.28204870e02,
    7.45742632e02,
    7.64600986e02,
    7.78296943e02,
    8.06798275e02,
    8.29269357e02,
    8.53281043e02,
    8.69479627e02,
    9.03735332e02,
    9.33070120e02,
    9.68744062e02,
    9.99168705e02,
    1.01714117e03,
    1.05949556e03,
    1.08715964e03,
    1.13783936e03,
    1.17969213e03,
    1.22507938e03,
    1.29387792e03,
    1.33968372e03,
    1.41408879e03,
    1.47787033e03,
    1.55296208e03,
    1.65777698e03,
    1.76772225e03,
    1.92133082e03,
    2.01266955e03,
    2.16339096e03,
    2.36594724e03,
    2.52971604e03,
    2.75235608e03,
    2.96505732e03,
    3.19875311e03,
    3.53155823e03,
    3.74790481e03,
    4.15373088e03,
    4.59732097e03,
    5.03516574e03,
    5.57194948e03,
    6.18417137e03,
    6.59712639e03,
    7.06936037e03,
    7.59323099e03,
    8.35899600e03,
    8.82944316e03,
    9.23694377e03,
    1.00177326e04,
    1.04208669e04,
    1.11105713e04,
    1.13412435e04,
    1.15315538e04,
    1.18929867e04,
    1.17798131e04,
    1.23884312e04,
    1.29485392e04,
    1.36207550e04,
    1.38288065e04,
    1.51347548e04,
    1.53811693e04,
    1.54029764e04,
    1.54847258e04,
    1.66567793e04,
    1.79101657e04,
    1.85923406e04,
    1.88601053e04,
    1.97313136e04,
    2.00628497e04,
    2.16301408e04,
    2.29609719e04,
    2.44373377e04,
    2.69059568e04,
    2.97021979e04,
    3.17818491e04,
    3.78057027e04,
    4.15539279e04,
    4.44437802e04,
    5.41338501e04,
    6.10211032e04,
    6.77450400e04,
    8.78940332e04,
    1.09288756e05,
    1.28961585e05,
    2.04704735e05,
    2.69213090e05,
    4.34566604e05,
    6.55038912e05,
    8.03969375e05,
    1.47310713e06,
    1.98718784e06,
    3.15746290e06,
    4.71815170e06,
    7.14871470e06,
    9.32967851e06,
    1.94276835e07,
    2.06419137e07,
    6.60541238e06,
    1.65135310e09,
    1.65135310e09,
    1.65135310e09,
]

test_pixel_count = np.array(
    [
        2.61608789e08,
        8.70589000e05,
        6.77175000e05,
        5.92301000e05,
        5.35265000e05,
        4.99780000e05,
        4.70842000e05,
        4.44128000e05,
        4.27749000e05,
        4.09271000e05,
        3.97814000e05,
        3.86751000e05,
        3.82678000e05,
        3.68068000e05,
        3.57058000e05,
        3.50907000e05,
        3.50034000e05,
        3.45386000e05,
        3.35622000e05,
        3.33727000e05,
        3.31449000e05,
        3.25465000e05,
        3.20527000e05,
        3.17095000e05,
        3.18708000e05,
        3.12437000e05,
        3.11527000e05,
        3.14896000e05,
        3.11130000e05,
        3.07054000e05,
        3.03590000e05,
        3.03176000e05,
        3.05771000e05,
        3.03643000e05,
        3.05324000e05,
        3.05293000e05,
        3.02469000e05,
        3.05703000e05,
        3.04541000e05,
        3.09389000e05,
        3.04141000e05,
        3.07382000e05,
        3.06714000e05,
        3.11372000e05,
        3.09251000e05,
        3.09336000e05,
        3.07980000e05,
        3.08018000e05,
        3.11976000e05,
        3.15690000e05,
        3.12731000e05,
        3.14875000e05,
        3.12652000e05,
        3.17953000e05,
        3.20004000e05,
        3.25671000e05,
        3.25042000e05,
        3.27401000e05,
        3.30497000e05,
        3.30112000e05,
        3.36985000e05,
        3.36305000e05,
        3.40664000e05,
        3.46152000e05,
        3.46063000e05,
        3.50798000e05,
        3.48291000e05,
        3.47882000e05,
        3.53673000e05,
        3.52344000e05,
        3.61070000e05,
        3.61520000e05,
        3.62854000e05,
        3.63065000e05,
        3.69348000e05,
        3.66894000e05,
        3.71571000e05,
        3.70721000e05,
        3.72996000e05,
        3.81050000e05,
        3.80328000e05,
        3.77435000e05,
        3.86687000e05,
        3.94999000e05,
        4.01563000e05,
        3.95119000e05,
        4.07603000e05,
        4.07214000e05,
        4.13005000e05,
        4.12543000e05,
        4.17785000e05,
        4.25076000e05,
        4.23794000e05,
        4.30603000e05,
        4.33600000e05,
        4.33820000e05,
        4.45709000e05,
        4.35901000e05,
        4.49961000e05,
        4.41873000e05,
        4.45922000e05,
        4.54132000e05,
        4.66204000e05,
        4.56113000e05,
        4.65354000e05,
        4.70254000e05,
        4.69964000e05,
        4.78939000e05,
        4.78007000e05,
        4.78390000e05,
        4.86355000e05,
        4.94415000e05,
        4.86425000e05,
        4.95141000e05,
        4.94601000e05,
        5.03923000e05,
        5.09084000e05,
        5.05347000e05,
        5.09583000e05,
        5.11916000e05,
        5.19265000e05,
        5.15211000e05,
        5.23362000e05,
        5.31719000e05,
        5.34259000e05,
        5.38921000e05,
        5.32217000e05,
        5.44701000e05,
        5.45095000e05,
        5.53303000e05,
        5.59511000e05,
        5.50710000e05,
        5.57207000e05,
        5.59545000e05,
        5.68677000e05,
        5.64924000e05,
        5.62459000e05,
        5.57846000e05,
        5.60422000e05,
        5.63504000e05,
        5.60417000e05,
        5.64767000e05,
        5.62293000e05,
        5.56363000e05,
        5.45148000e05,
        5.46342000e05,
        5.39931000e05,
        5.42966000e05,
        5.34930000e05,
        5.29854000e05,
        5.22334000e05,
        5.22982000e05,
        5.17273000e05,
        5.06943000e05,
        5.01069000e05,
        4.90343000e05,
        4.87575000e05,
        4.87800000e05,
        4.72277000e05,
        4.64653000e05,
        4.50038000e05,
        4.40220000e05,
        4.33069000e05,
        4.15232000e05,
        4.07058000e05,
        3.89329000e05,
        3.83028000e05,
        3.76729000e05,
        3.63625000e05,
        3.55969000e05,
        3.39469000e05,
        3.28632000e05,
        3.17953000e05,
        3.02713000e05,
        2.93171000e05,
        2.84139000e05,
        2.69375000e05,
        2.52347000e05,
        2.42367000e05,
        2.32992000e05,
        2.17551000e05,
        2.10769000e05,
        2.00831000e05,
        1.87867000e05,
        1.82706000e05,
        1.78255000e05,
        1.66477000e05,
        1.64702000e05,
        1.51130000e05,
        1.48167000e05,
        1.39763000e05,
        1.34129000e05,
        1.32969000e05,
        1.21606000e05,
        1.20079000e05,
        1.18996000e05,
        1.16119000e05,
        1.12291000e05,
        1.05205000e05,
        1.00231000e05,
        9.99940000e04,
        9.56680000e04,
        9.53290000e04,
        9.30400000e04,
        9.30720000e04,
        9.23140000e04,
        9.20520000e04,
        9.25080000e04,
        8.90580000e04,
        8.77800000e04,
        8.61990000e04,
        8.14860000e04,
        7.93220000e04,
        7.81970000e04,
        7.76620000e04,
        7.77700000e04,
        7.39080000e04,
        6.80710000e04,
        6.59990000e04,
        6.27870000e04,
        6.14030000e04,
        6.17370000e04,
        5.92130000e04,
        5.58160000e04,
        5.22610000e04,
        4.71610000e04,
        4.11220000e04,
        3.68130000e04,
        3.08360000e04,
        2.77450000e04,
        2.56110000e04,
        2.01160000e04,
        1.77100000e04,
        1.61200000e04,
        1.22440000e04,
        9.56600000e03,
        7.92400000e03,
        4.16300000e03,
        3.17700000e03,
        1.98600000e03,
        1.18200000e03,
        9.60000000e02,
        2.13000000e02,
        2.04000000e02,
        9.50000000e01,
        3.50000000e01,
        8.00000000e00,
        0.00000000e00,
        0.00000000e00,
        3.50000000e01,
        2.45000000e02,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)

train_pixel_count = np.array(
    [
        1.38974431e09,
        6.48999100e06,
        5.05640700e06,
        4.37776400e06,
        3.95679300e06,
        3.65414200e06,
        3.43712400e06,
        3.25330100e06,
        3.11546000e06,
        3.00674800e06,
        2.90293600e06,
        2.81899400e06,
        2.75146300e06,
        2.69185400e06,
        2.64429400e06,
        2.58473900e06,
        2.54483300e06,
        2.51058300e06,
        2.48386000e06,
        2.45217900e06,
        2.44036500e06,
        2.41106200e06,
        2.39486300e06,
        2.38139900e06,
        2.35973000e06,
        2.35241700e06,
        2.33992800e06,
        2.32583200e06,
        2.32379500e06,
        2.31790500e06,
        2.31435400e06,
        2.31260400e06,
        2.31078200e06,
        2.30060800e06,
        2.31390800e06,
        2.31833000e06,
        2.33381200e06,
        2.32939900e06,
        2.34279200e06,
        2.33958800e06,
        2.35042000e06,
        2.36488600e06,
        2.38346500e06,
        2.39155100e06,
        2.40209100e06,
        2.39874400e06,
        2.42086700e06,
        2.42941500e06,
        2.44683600e06,
        2.46428000e06,
        2.47682600e06,
        2.48856300e06,
        2.49726500e06,
        2.51646500e06,
        2.53810500e06,
        2.56422800e06,
        2.56010000e06,
        2.57834200e06,
        2.61011300e06,
        2.61941300e06,
        2.64119400e06,
        2.66608100e06,
        2.67944100e06,
        2.70082600e06,
        2.72527900e06,
        2.75045900e06,
        2.75925700e06,
        2.79327000e06,
        2.82398800e06,
        2.83871700e06,
        2.86524700e06,
        2.87194700e06,
        2.90373900e06,
        2.92247200e06,
        2.94352800e06,
        2.95561800e06,
        2.97470300e06,
        2.98959500e06,
        3.00022800e06,
        3.02987100e06,
        3.03900200e06,
        3.05922900e06,
        3.05777500e06,
        3.09481200e06,
        3.10975800e06,
        3.11836000e06,
        3.12806400e06,
        3.14533800e06,
        3.14105200e06,
        3.14483300e06,
        3.15634900e06,
        3.16566600e06,
        3.19167300e06,
        3.18935700e06,
        3.18282700e06,
        3.19531000e06,
        3.19327800e06,
        3.21916900e06,
        3.20518900e06,
        3.20122600e06,
        3.19346500e06,
        3.19682300e06,
        3.18662800e06,
        3.17938200e06,
        3.19894700e06,
        3.17253700e06,
        3.19680700e06,
        3.19508800e06,
        3.17229200e06,
        3.18210300e06,
        3.15610300e06,
        3.16643200e06,
        3.14919700e06,
        3.13987500e06,
        3.12587700e06,
        3.10680200e06,
        3.09446200e06,
        3.07691400e06,
        3.04093100e06,
        3.02335700e06,
        3.00870500e06,
        2.96955800e06,
        2.93415700e06,
        2.91517400e06,
        2.90679400e06,
        2.87994300e06,
        2.85000300e06,
        2.81419500e06,
        2.77862000e06,
        2.75073000e06,
        2.73278100e06,
        2.68701700e06,
        2.65855900e06,
        2.62424300e06,
        2.59686900e06,
        2.53829800e06,
        2.51665000e06,
        2.47872400e06,
        2.45084000e06,
        2.42807900e06,
        2.38255700e06,
        2.35247200e06,
        2.30395100e06,
        2.26324500e06,
        2.24433000e06,
        2.20261400e06,
        2.16615500e06,
        2.13389500e06,
        2.07632600e06,
        2.02627800e06,
        1.98814300e06,
        1.96086500e06,
        1.90388800e06,
        1.86642800e06,
        1.83082100e06,
        1.77736100e06,
        1.72679900e06,
        1.67195800e06,
        1.64947500e06,
        1.58214500e06,
        1.54129700e06,
        1.49507800e06,
        1.46617400e06,
        1.41202100e06,
        1.36274800e06,
        1.31530400e06,
        1.26969900e06,
        1.24679500e06,
        1.19499700e06,
        1.16299200e06,
        1.11183700e06,
        1.07118500e06,
        1.03000300e06,
        9.73569000e05,
        9.39473000e05,
        8.83647000e05,
        8.48012000e05,
        8.11010000e05,
        7.53758000e05,
        7.01178000e05,
        6.41933000e05,
        6.09710000e05,
        5.62486000e05,
        5.10100000e05,
        4.70076000e05,
        4.21723000e05,
        3.90461000e05,
        3.51547000e05,
        3.16469000e05,
        2.92440000e05,
        2.57796000e05,
        2.25070000e05,
        1.94995000e05,
        1.74763000e05,
        1.46950000e05,
        1.31318000e05,
        1.17474000e05,
        1.05186000e05,
        9.23490000e04,
        8.67970000e04,
        7.87830000e04,
        6.91750000e04,
        6.31370000e04,
        5.55890000e04,
        5.25340000e04,
        5.08890000e04,
        4.67990000e04,
        4.76770000e04,
        4.42400000e04,
        3.97520000e04,
        3.50390000e04,
        3.79280000e04,
        2.97880000e04,
        2.91650000e04,
        2.95480000e04,
        2.88740000e04,
        2.52320000e04,
        2.41310000e04,
        2.28200000e04,
        2.47710000e04,
        2.22890000e04,
        2.05720000e04,
        1.71320000e04,
        1.61040000e04,
        1.53140000e04,
        1.42140000e04,
        1.44750000e04,
        1.51460000e04,
        1.28440000e04,
        1.19950000e04,
        1.15450000e04,
        1.03890000e04,
        9.35200000e03,
        8.25600000e03,
        6.54400000e03,
        5.54400000e03,
        4.88100000e03,
        3.90400000e03,
        2.95700000e03,
        1.81400000e03,
        1.33900000e03,
        1.09400000e03,
        9.08000000e02,
        6.27000000e02,
        4.28000000e02,
        3.15000000e02,
        2.23000000e02,
        1.77000000e02,
        8.50000000e01,
        4.50000000e01,
        5.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)


all_pixel_count = [
    1.6513531e09,
    7.3605800e06,
    5.7335820e06,
    4.9700650e06,
    4.4920580e06,
    4.1539220e06,
    3.9079660e06,
    3.6974290e06,
    3.5432090e06,
    3.4160190e06,
    3.3007500e06,
    3.2057450e06,
    3.1341410e06,
    3.0599220e06,
    3.0013520e06,
    2.9356460e06,
    2.8948670e06,
    2.8559690e06,
    2.8194820e06,
    2.7859060e06,
    2.7718140e06,
    2.7365270e06,
    2.7153900e06,
    2.6984940e06,
    2.6784380e06,
    2.6648540e06,
    2.6514550e06,
    2.6407280e06,
    2.6349250e06,
    2.6249590e06,
    2.6179440e06,
    2.6157800e06,
    2.6165530e06,
    2.6042510e06,
    2.6192320e06,
    2.6236230e06,
    2.6362810e06,
    2.6351020e06,
    2.6473330e06,
    2.6489770e06,
    2.6545610e06,
    2.6722680e06,
    2.6901790e06,
    2.7029230e06,
    2.7113420e06,
    2.7080800e06,
    2.7288470e06,
    2.7374330e06,
    2.7588120e06,
    2.7799700e06,
    2.7895570e06,
    2.8034380e06,
    2.8099170e06,
    2.8344180e06,
    2.8581090e06,
    2.8898990e06,
    2.8851420e06,
    2.9057430e06,
    2.9406100e06,
    2.9495250e06,
    2.9781790e06,
    3.0023860e06,
    3.0201050e06,
    3.0469780e06,
    3.0713420e06,
    3.1012570e06,
    3.1075480e06,
    3.1411520e06,
    3.1776610e06,
    3.1910610e06,
    3.2263170e06,
    3.2334670e06,
    3.2665930e06,
    3.2855370e06,
    3.3128760e06,
    3.3225120e06,
    3.3462740e06,
    3.3603160e06,
    3.3732240e06,
    3.4109210e06,
    3.4193300e06,
    3.4366640e06,
    3.4444620e06,
    3.4898110e06,
    3.5113210e06,
    3.5134790e06,
    3.5356670e06,
    3.5525520e06,
    3.5540570e06,
    3.5573760e06,
    3.5741340e06,
    3.5907420e06,
    3.6154670e06,
    3.6199600e06,
    3.6164270e06,
    3.6291300e06,
    3.6389870e06,
    3.6550700e06,
    3.6551500e06,
    3.6430990e06,
    3.6393870e06,
    3.6509550e06,
    3.6528320e06,
    3.6354950e06,
    3.6643010e06,
    3.6427910e06,
    3.6667710e06,
    3.6740270e06,
    3.6502990e06,
    3.6604930e06,
    3.6424580e06,
    3.6608470e06,
    3.6356220e06,
    3.6350160e06,
    3.6204780e06,
    3.6107250e06,
    3.6035460e06,
    3.5822610e06,
    3.5505140e06,
    3.5352730e06,
    3.5279700e06,
    3.4847690e06,
    3.4575190e06,
    3.4468930e06,
    3.4410530e06,
    3.4188640e06,
    3.3822200e06,
    3.3588960e06,
    3.3237150e06,
    3.3040330e06,
    3.2922920e06,
    3.2377270e06,
    3.2157660e06,
    3.1837880e06,
    3.1655460e06,
    3.1032220e06,
    3.0791090e06,
    3.0365700e06,
    3.0112620e06,
    2.9915830e06,
    2.9429740e06,
    2.9172390e06,
    2.8662440e06,
    2.8196080e06,
    2.7894780e06,
    2.7489560e06,
    2.7060860e06,
    2.6768610e06,
    2.6112560e06,
    2.5561320e06,
    2.5104770e06,
    2.4838470e06,
    2.4211610e06,
    2.3733710e06,
    2.3318900e06,
    2.2677040e06,
    2.2143740e06,
    2.1597580e06,
    2.1217520e06,
    2.0467980e06,
    1.9913350e06,
    1.9352980e06,
    1.8992430e06,
    1.8272530e06,
    1.7698060e06,
    1.7046330e06,
    1.6527270e06,
    1.6235240e06,
    1.5586220e06,
    1.5189610e06,
    1.4513060e06,
    1.3998170e06,
    1.3479560e06,
    1.2762820e06,
    1.2326440e06,
    1.1677860e06,
    1.1173870e06,
    1.0633570e06,
    9.9612500e05,
    9.3417000e05,
    8.5948400e05,
    8.2047900e05,
    7.6331700e05,
    6.9796700e05,
    6.5278200e05,
    5.9997800e05,
    5.5693800e05,
    5.1624900e05,
    4.6759900e05,
    4.4060700e05,
    3.9755900e05,
    3.5919900e05,
    3.2796400e05,
    2.9636900e05,
    2.6702900e05,
    2.5031400e05,
    2.3359300e05,
    2.1747700e05,
    1.9755400e05,
    1.8702800e05,
    1.7877700e05,
    1.6484300e05,
    1.5846600e05,
    1.4862900e05,
    1.4560600e05,
    1.4320300e05,
    1.3885100e05,
    1.4018500e05,
    1.3329800e05,
    1.2753200e05,
    1.2123800e05,
    1.1941400e05,
    1.0911000e05,
    1.0736200e05,
    1.0721000e05,
    1.0664400e05,
    9.9140000e04,
    9.2202000e04,
    8.8819000e04,
    8.7558000e04,
    8.3692000e04,
    8.2309000e04,
    7.6345000e04,
    7.1920000e04,
    6.7575000e04,
    6.1375000e04,
    5.5597000e04,
    5.1959000e04,
    4.3680000e04,
    3.9740000e04,
    3.7156000e04,
    3.0505000e04,
    2.7062000e04,
    2.4376000e04,
    1.8788000e04,
    1.5110000e04,
    1.2805000e04,
    8.0670000e03,
    6.1340000e03,
    3.8000000e03,
    2.5210000e03,
    2.0540000e03,
    1.1210000e03,
    8.3100000e02,
    5.2300000e02,
    3.5000000e02,
    2.3100000e02,
    1.7700000e02,
    8.5000000e01,
    8.0000000e01,
    2.5000000e02,
    0.0000000e00,
    0.0000000e00,
    0.0000000e00,
]
