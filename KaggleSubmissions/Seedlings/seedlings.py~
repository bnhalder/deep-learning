from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = '/home/paperspace/data/seedlings/'
df = pd.DataFrame(columns=["file", "species"])
for image in glob(f'{PATH}train/**/*.png'):
    dir_ = image.split('/')
    file_, species = dir_[-1], dir_[-2]
    df = df.append({"file": file_, "species": species.replace(" ", "_")}, ignore_index=True)
df.to_csv(f'{PATH}labels.csv', index=False)
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)

def get_data(sz, bs):
    tfms = tfms_from_model(f_model, sz, aug_tfms = transforms_side_on, max_zoom = 1.05)
    return ImageClassifierData.from_csv(PATH, 'train-all', label_csv, tfms=tfms, val_idxs=val_idxs, test_name='test', bs=bs)

sz = 224
bs = 16
f_model = resnet50
data = get_data(sz, bs)
learn = ConvLearner.pretrained(f_model, data, precompute = True)

#lrf=learn.lr_find()
#learn.sched.plot()
lr = 0.01
learn.fit(lr, 3, cycle_len=1)
learn.precompute=False
lrs = np.array([lr/9, lr/3, lr])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len = 1, cycle_mult = 2)
learn.save(f'{sz}')

sz = 300
bs = 16
learn.set_data(get_data(sz, bs))
learn.freeze()
learn.fit(lr, 3, cycle_len = 1, cycle_mult=2)
lrs = np.array([lr/18, lr/9, lr/3])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')

#Prepare submission file without TTA
log_preds_test = learn.predict(is_test=True)
log_preds_test = np.argmax(log_preds_test, axis=1)
preds_classes = [data.classes[i].replace("_", " ") for i in log_preds_test]
probs = np.exp(log_preds_test)
submission = pd.DataFrame({'file': os.listdir(f'{PATH}test'), 'species': preds_classes})
submission.to_csv(f'{PATH}submission_13_02_2018_01.csv', index=False)

#prepare submission file with TTA
#tta = learn.TTA(is_test = True)
#mean_logpreds = np.mean(tta[0], 0)
#max_preds = np.argmax(mean_logpreds, 1)
#preds_classes = [data.classes[i].replace("_", " ") for i in log_preds_test]
#submission = pd.DataFrame({'file': os.listdir(f'{PATH}test'), 'species': preds_classes})
#submission.to_csv(f'{PATH}submission_13_02_2018_02.csv', index=False)




