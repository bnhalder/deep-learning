from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = '/home/paperspace/data/dogbreeds/'

f_model = resnext101_64
label_csv = f"{PATH}labels.csv"
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)
bs = 24
sz = 224

def get_data(sz, bs):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_csv(PATH, 'train' ,f'{PATH}labels.csv', test_name='test', 
			val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
    return data if sz > 300 else data.resize(340, 'tmp')

data = get_data(sz, bs)
learn = ConvLearner.pretrained(f_model, data, precompute=True)

#lrf = learn.lr_find()
#learn.sched.plot()

lr = 0.01
learn.fit(lr, 5)

from sklearn import metrics
data = get_data(sz, bs)
learn = ConvLearner.pretrained(f_model, data, precompute=True, ps = 0.5)
learn.fit(lr, 2)

learn.precompute = False
learn.fit(lr, 5, cycle_len = 1)

learn.save('224_pre')
learn.load('224_pre')

learn.set_data(get_data(299, bs))
learn.freeze()
learn.fit(0.01, 3, cycle_len=1)

learn.fit(0.01, 1, cycle_len=2)
learn.fit(0.01, 1, cycle_len=2)

learn.set_data(get_data(350, bs))
learn.freeze()
learn.fit(0.01, 3, cycle_len=1, cycle_mult=2)
learn.unfreeze()
lrs = np.array([lr/10, lr/5, lr/2])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')

log_preds, y = learn.TTA(is_test = True)
preds = np.mean(log_preds, 0)
probs = np.exp(preds)
ds = pd.DataFrame(probs)
ds.columns = data.classes
ds.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])
ds.to_csv(PATH+"Degbreeds_submission_2018_02_13_02.csv", index=False)


