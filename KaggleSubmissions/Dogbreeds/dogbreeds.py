from fastai.conv_learner import *
PATH = '/home/paperspace/data/dogbreeds/'
f_model = resnext101_64
label_csv = f"{PATH}labels.csv"
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)

def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms = transforms_side_on, max_zoom = 1.05)
    return ImageClassifierData.from_csv(PATH, 'train', label_csv, tfms=tfms, suffix='.jpg', val_idxs=val_idxs, test_name='test')

sz = 256
data = get_data(256)
learn = ConvLearner.pretrained(f_model, data, precompute = True)

lrf = learn.lr_find()
learn.sched.plot()

lr = 0.01
learn.fit(lr, 3, cycle_len = 1, cycle_mult = 2)

learn.precompute = False
learn.fit(lr, 3, cycle_len = 1, cycle_mult = 2)
lrs = np.array([lr/9, lr/3, lr])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len = 1, cycle_mult = 2)
learn.save(f'{sz}')

sz = 300
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len = 1, cycle_mult=2)
learn.unfreeze()
lrs = lrs = np.array([lr/20, lr/10, lr/2])
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')

log_preds, y = learn.TTA(is_test = True)
preds = np.mean(log_preds, 0)
probs = np.exp(preds)
ds = pd.DataFrame(probs)
ds.columns = data.classes
ds.insert(0, 'id', [o[5:-4] for o in data.test_ds.fnames])
ds.to_csv(PATH+"Degbreeds_submission_2018_02_13_01.csv", index=False)


