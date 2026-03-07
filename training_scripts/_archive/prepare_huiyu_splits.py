import glob, os, random, json

random.seed(42)

asd_files = sorted(glob.glob(
    '/home/mcasu/HD/max_project/huiyu_2019_eye_movements/TrainingData/ASD/*.txt'))
td_files  = sorted(glob.glob(
    '/home/mcasu/HD/max_project/huiyu_2019_eye_movements/TrainingData/TD/*.txt'))

# Estrai soggetti unici dal filename (ASD_scanpath_1.txt → "ASD_1")
def subj_from_path(p):
    b = os.path.basename(p).replace('.txt','')
    parts = b.split('_')
    return f"{parts[0]}_{parts[-1]}"  # es. "ASD_1"

asd_subjs = sorted(set(subj_from_path(f) for f in asd_files))
td_subjs  = sorted(set(subj_from_path(f) for f in td_files))

print(f"ASD soggetti: {len(asd_subjs)}, TD soggetti: {len(td_subjs)}")

# Holdout: 2 ASD + 2 TD (mai visti in Phase 1)
holdout_asd = random.sample(asd_subjs, 2)
holdout_td  = random.sample(td_subjs, 2)
holdout = holdout_asd + holdout_td
finetune = [s for s in asd_subjs + td_subjs if s not in holdout]

splits = {'holdout': holdout, 'finetune': finetune}
with open('/home/mcasu/HD/max_project/huiyu_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)

print(f"Holdout ({len(holdout)}): {holdout}")
print(f"Fine-tune ({len(finetune)}): {finetune}")
print("Salvato: huiyu_splits.json")
