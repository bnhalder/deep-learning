from fastai.conv_learner import *
PATH = '/home/paperspace/data/planet/'
from fastai.plots import *
from planet import f2

metrics = [f2]
f_model = resnet34
label_csv = f"{PATH}train_v2.csv"

n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)

def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms = transforms_top_down, max_zoom = 1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms, suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')

#First run on image size 64x64
sz = 64
data = get_data(sz)
data = data.resize(int(sz*1.3), 'tmp')
learn = ConvLearner.pretrained(f_model, data, metrics=metrics)

lr = 0.2
learn.fit(lr, 3, cycle_len = 1, cycle_mult = 2)
lrs = np.array([lr/9, lr/3, lr])
learn.unfreeze()
learn.fit(lrs, 3, cycle_len = 1, cycle_mult = 2)
learn.save(f'{sz}')

#Train on image size 128x128
sz = 128
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len = 1, cycle_mult=2)
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')

#Train on image size 256x256
sz = 256
learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len = 1, cycle_mult=2)
learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')

#Evaluate test dataset
tta = learn.TTA(is_test = True)

#Create submission file
test_fnames = data.test_ds.fnames
for i in range(len(test_fnames)):
    test_fnames[i] = test_fnames[i].split("/")[1].split(".")[0]
classes = np.array(data.classes, dtype=str)
preds = np.mean(tta[0], 0)
res = [" ".join(classes[np.where(pp > 0.2)]) for pp in preds] 
submission = pd.DataFrame(data=res)
submission.columns = ["tags"]
submission.insert(0, 'image_name', test_fnames)
submission.to_csv(PATH+"Planet_submission_2018_02_12_01.csv", index=False)

